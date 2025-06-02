import cv2
import numpy as np
import dlib
import time
from imutils import face_utils

# =========================
# 1. Inicializar modelos y parámetros
# =========================

# 1.1 Reconocimiento LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

# 1.2 Clasificadores Haar (frontal y perfil)
face_frontal = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_perfil  = cv2.CascadeClassifier('haarcascade_profileface.xml')

# 1.3 Detector y predictor Dlib (landmarks)
detector_dlib   = dlib.get_frontal_face_detector()
predictor_dlib  = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 1.4 Nombres de usuarios entrenados (ID coincide con índice LBPH)
usuarios = ["Jose", "Marcelo", "Samuel", "Yoana"]

# 1.5 Parámetros de umbral y tiempos
umbral_reconocimiento = 70    # LBPH: confianza < umbral → candidato registrado
life_timeout         = 5.0    # segundos para detectar parpadeo
mov_thresh           = 15     # píxeles mínimos de desplazamiento de cabeza
min_screen_area      = 15000  # área mínima de contorno para considerar posible pantalla
screen_ar_tolerance  = 0.3    # tolerancia de aspecto (ancho/alto) para una pantalla de móvil (~0.56–0.75)

# =========================
# 2. Variables de estado global
# =========================

active_faces = {}
# key = (fx, fy, usuario_id)
# valor = {
#   "start_time": float,
#   "blink_count": int,
#   "last_blink_time": float,
#   "positions": [(cx, cy), ...]
# }

# =========================
# 3. Funciones auxiliares
# =========================

def detectar_parpadeo(shape_np):
    ojo_izq = shape_np[36:42]
    ojo_der = shape_np[42:48]
    ratio_izq = np.linalg.norm(ojo_izq[1] - ojo_izq[5]) / np.linalg.norm(ojo_izq[0] - ojo_izq[3])
    ratio_der = np.linalg.norm(ojo_der[1] - ojo_der[5]) / np.linalg.norm(ojo_der[0] - ojo_der[3])
    ear = (ratio_izq + ratio_der) / 2.0
    return ear < 0.18

def max_desplazamiento(positions):
    if not positions:
        return 0.0
    x0, y0 = positions[0]
    max_dist = 0.0
    for (x, y) in positions:
        dist = np.hypot(x - x0, y - y0)
        if dist > max_dist:
            max_dist = dist
    return max_dist

def detectar_pantallas(frame_gray):
    """
    Detecta contornos rectangulares que puedan corresponder
    a pantallas de móvil. Retorna lista de rectángulos (x, y, w, h).
    """
    # 1. Aplicar desenfoque ligero y Canny
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # 2. Encontrar contornos
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pantallas = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_screen_area:
            continue
        # Aproximar polígono
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            # Cuadrilátero hallado, extraer bounding rect
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            # Aspect ratio típico de móviles (aprox. 9:16 = 0.56, 16:9 = 1.78). Queremos vertical:
            # consideramos pantallas con AR en rango [0.4, 0.8]
            if 0.4 <= ar <= 0.8:
                pantallas.append((x, y, w, h))
    return pantallas

# =========================
# 4. Bucle principal
# =========================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar + espejo
    frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux = frame.copy()
    ahora = time.time()

    # Detectar posibles pantallas en escala de grises
    pantallas = detectar_pantallas(gray)
    # Dibujar rectángulos de pantallas detectadas (debug visual)
    for (sx, sy, sw, sh) in pantallas:
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)

    # Limpiar active_faces expirados
    to_rm = []
    for key, data in active_faces.items():
        if ahora - data["start_time"] > (life_timeout + 1.0):
            to_rm.append(key)
    for k in to_rm:
        del active_faces[k]

    # Detectar rostros Haar
    faces = face_frontal.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(60, 60))
    if len(faces) == 0:
        faces = face_perfil.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    rects_dlib = detector_dlib(gray, 0)

    resultados = []

    for (x, y, w, h) in faces:
        cx, cy = x + w//2, y + h//2
        roi_color = aux[y:y+h, x:x+w]
        roi_gray  = cv2.cvtColor(cv2.resize(roi_color, (150, 150)), cv2.COLOR_BGR2GRAY)

        texto = ""
        color = (255, 255, 255)

        id_usuario, conf = face_recognizer.predict(roi_gray)
        es_registrado = (conf < umbral_reconocimiento and id_usuario < len(usuarios))

        # Verificar si rostro está dentro de alguna pantalla detectada
        en_pantalla = False
        for (sx, sy, sw, sh) in pantallas:
            if sx < cx < sx+sw and sy < cy < sy+sh:
                en_pantalla = True
                break

        if es_registrado and not en_pantalla:
            # Desafío liveness + movimiento
            fx = int(round(cx/50.0)*50)
            fy = int(round(cy/50.0)*50)
            key = (fx, fy, id_usuario)
            if key not in active_faces:
                active_faces[key] = {
                    "start_time": ahora,
                    "blink_count": 0,
                    "last_blink_time": 0.0,
                    "positions": [(cx, cy)]
                }
            else:
                active_faces[key]["positions"].append((cx, cy))

            state = active_faces[key]
            elapsed = ahora - state["start_time"]

            # Buscar rect Dlib para landmarks
            rect_l = None
            for rect in rects_dlib:
                rx, ry = rect.left(), rect.top()
                rw, rh = rect.width(), rect.height()
                if (x < rx < x+w and y < ry < y+h) or (rx < x < rx+rw and ry < y < ry+rh):
                    rect_l = rect
                    break

            if rect_l is not None:
                shape = predictor_dlib(gray, rect_l)
                shape_np = face_utils.shape_to_np(shape)
                if detectar_parpadeo(shape_np) and (ahora - state["last_blink_time"] > 0.5):
                    state["blink_count"] += 1
                    state["last_blink_time"] = ahora

            if state["blink_count"] >= 1:
                desplaz = max_desplazamiento(state["positions"])
                if desplaz >= mov_thresh:
                    nombre = usuarios[id_usuario]
                    texto  = f"{nombre} - Bienvenido"
                    color  = (0, 255, 0)
                    # Reiniciar estado
                    state["start_time"]      = ahora
                    state["blink_count"]     = 0
                    state["last_blink_time"] = ahora
                    state["positions"]       = [(cx, cy)]
                else:
                    texto = "¡Spoofing de vídeo!"
                    color = (0, 0, 255)
                    del active_faces[key]
            elif elapsed >= life_timeout:
                texto = "¡Spoofing de vídeo!"
                color = (0, 0, 255)
                del active_faces[key]
            else:
                texto = "Parpadea para verificar..."
                color = (255, 255, 255)

        elif es_registrado and en_pantalla:
            # Si la cara registrada está dentro de la pantalla detectada → spoofing
            texto = "¡Spoofing de vídeo!"
            color = (0, 0, 255)
        else:
            # No registrado → lógica de foto estática
            rect_l = None
            for rect in rects_dlib:
                rx, ry = rect.left(), rect.top()
                rw, rh = rect.width(), rect.height()
                if (x < rx < x+w and y < ry < y+h) or (rx < x < rx+rw and ry < y < ry+rh):
                    rect_l = rect
                    break

            fx = int(round(cx/50.0)*50)
            fy = int(round(cy/50.0)*50)
            key_unk = (fx, fy)
            if key_unk not in active_faces:
                active_faces[key_unk] = {"start_time": ahora, "blinked": False}
            state_unk   = active_faces[key_unk]
            elapsed_unk = ahora - state_unk["start_time"]

            if rect_l is not None:
                shape = predictor_dlib(gray, rect_l)
                shape_np = face_utils.shape_to_np(shape)
                if detectar_parpadeo(shape_np):
                    state_unk["blinked"] = True

            if state_unk["blinked"]:
                texto = "Desconocido"
                color = (0, 0, 255)
                del active_faces[key_unk]
            elif elapsed_unk >= life_timeout:
                texto = "Imagen estática"
                color = (0, 128, 255)
                del active_faces[key_unk]
            else:
                texto = "Parpadea para verificar..."
                color = (255, 255, 255)

        resultados.append((x, y, w, h, texto, color))

    # Dibujar resultados
    for (x, y, w, h, texto, color) in resultados:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            texto,
            (x, max(y-10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Reconocimiento Facial - Video Spoofing", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

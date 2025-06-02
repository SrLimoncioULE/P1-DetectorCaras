import cv2
import numpy as np
import dlib
import time
from imutils import face_utils

# =========================
# 1. Inicializar modelos y parámetros
# =========================

# Reconocimiento LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Models/modeloLBPHFace.xml')  # Ajusta la ruta si hace falta

# Clasificadores Haar (frontal y perfil)
face_frontal = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
face_perfil = cv2.CascadeClassifier('Cascades/haarcascade_profileface.xml')

# Detector y predictor Dlib (landmarks)
detector_dlib = dlib.get_frontal_face_detector()
predictor_dlib = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Lista de nombres de usuarios entrenados (orden debe coincidir con etiquetas en el entrenamiento)
usuarios = ["Jose", "Marcelo", "Samuel", "Yoana"]

# Umbral de confianza LBPH: solo se considera “reconocido” si confianza < umbral
umbral_reconocimiento = 70  # Ajusta según tus pruebas

# Tiempo máximo que mantenemos a un usuario “confirmado” (en segundos)
tiempo_validez = 10.0

# Para prueba de liveness: ventana máxima para detectar un parpadeo
life_timeout = 5.0  # segundos

# =========================
# 2. Variables de estado global
# =========================

# Para almacenar usuarios reconocidos recientemente
usuarios_confirmados = []  
# Cada entrada: {"pos": (cx, cy), "nombre": str, "confianza": float, "timestamp": float}

# Para almacenar caras no reconocidas y su estado de parpadeo
# key = (fx, fy) : (posición aproximada de la cara), value = {"start_time": float, "blinked": bool}
active_faces = {}

# =========================
# 3. Funciones auxiliares
# =========================

def detectar_parpadeo(shape_np):
    """
    Recibe array Nx2 de los 68 landmarks de la cara (shape_np).
    Calcula el EAR (Eye Aspect Ratio) para cada ojo y devuelve True si
    promedia por debajo de un umbral (indicando parpadeo).
    """
    ojo_izq = shape_np[36:42]
    ojo_der = shape_np[42:48]
    # Distancia vertical (puntos 1-5) dividido por horizontal (0-3)
    ratio_izq = np.linalg.norm(ojo_izq[1] - ojo_izq[5]) / np.linalg.norm(ojo_izq[0] - ojo_izq[3])
    ratio_der = np.linalg.norm(ojo_der[1] - ojo_der[5]) / np.linalg.norm(ojo_der[0] - ojo_der[3])
    ear = (ratio_izq + ratio_der) / 2.0
    return ear < 0.18  # Umbral típico, ajustable si hace falta

def encontrar_usuario_existente(cx, cy, tiempo_actual):
    """
    Comprueba si ya tenemos un usuario confirmado cerca de (cx, cy) y no caducado.
    Si lo encuentra, devuelve la entrada; si no, devuelve None.
    """
    for usuario in usuarios_confirmados:
        ux, uy = usuario["pos"]
        dist = np.hypot(ux - cx, uy - cy)
        if dist < 50 and (tiempo_actual - usuario["timestamp"] < tiempo_validez):
            return usuario
    return None

# =========================
# 4. Bucle principal de captura y procesamiento
# =========================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar y espejar para que sea tipo “selfie”
    frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()
    tiempo_actual = time.time()

    # ------ 4.1 Limpiar active_faces que hayan expirado ------
    to_delete = []
    for face_key, data in active_faces.items():
        if tiempo_actual - data["start_time"] > (life_timeout + 1.0):
            to_delete.append(face_key)
    for k in to_delete:
        del active_faces[k]

    # ------ 4.2 Detectar rostros con Haar (frontal o perfil) ------
    faces = face_frontal.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(60, 60))
    if len(faces) == 0:
        faces = face_perfil.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

    # ------ 4.3 Detectar rectángulos Dlib para landmarks ------
    rects_dlib = detector_dlib(gray, 0)

    resultados = []

    # ------ 4.4 Procesar cada cara detectada por Haar ------
    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        rostro_color = aux_frame[y:y+h, x:x+w]
        rostro_gray = cv2.cvtColor(cv2.resize(rostro_color, (150, 150)), cv2.COLOR_BGR2GRAY)

        texto = ""
        color = (255, 255, 255)  # Por defecto, blanco (intermedio)

        # ---- 4.4.1 Intentar reconocer con LBPH ----
        id_usuario, confianza = face_recognizer.predict(rostro_gray)
        if confianza < umbral_reconocimiento and id_usuario < len(usuarios):
            # --- Rostro reconocido como usuario existente ---
            nombre = usuarios[id_usuario]
            texto = f"{nombre} - {confianza:.2f}"
            color = (0, 255, 0)  # Verde

            # Actualizar o agregar a usuarios_confirmados
            usuario_existente = encontrar_usuario_existente(cx, cy, tiempo_actual)
            if not usuario_existente:
                usuarios_confirmados.append({
                    "pos": (cx, cy),
                    "nombre": nombre,
                    "confianza": confianza,
                    "timestamp": tiempo_actual
                })
            else:
                # Si ya existía, actualizamos timestamp y confianza
                usuario_existente["timestamp"] = tiempo_actual
                usuario_existente["confianza"] = confianza

        else:
            # --- Rostro NO reconocido o confianza demasiado alta ---
            # Buscamos rectángulo Dlib asociado, para landmarks
            rect_asociado = None
            for rect in rects_dlib:
                rx, ry = rect.left(), rect.top()
                rw, rh = rect.width(), rect.height()
                # Comprobar solapamiento aproximado
                if (x < rx < x + w and y < ry < y + h) or (rx < x < rx + rw and ry < y < ry + rh):
                    rect_asociado = rect
                    break

            # Generar una clave aproximada para la cara, agrupando por posición
            fx = int(round(cx / 50.0) * 50)
            fy = int(round(cy / 50.0) * 50)
            face_key = (fx, fy)

            # Si es la primera vez que vemos esta key, inicializamos
            if face_key not in active_faces:
                active_faces[face_key] = {
                    "start_time": tiempo_actual,
                    "blinked": False
                }
            state = active_faces[face_key]
            elapsed = tiempo_actual - state["start_time"]

            # Si tenemos rect_asociado, extraemos landmarks y comprobamos parpadeo
            if rect_asociado is not None:
                shape = predictor_dlib(gray, rect_asociado)
                shape_np = face_utils.shape_to_np(shape)
                if detectar_parpadeo(shape_np):
                    state["blinked"] = True

            # Decidir etiqueta final según estado de parpadeo y tiempo transcurrido
            if state["blinked"]:
                texto = "Desconocido"
                color = (0, 0, 255)  # Rojo
            elif elapsed >= life_timeout:
                texto = "Imagen detectada"
                color = (0, 128, 255)  # Naranja
            else:
                texto = "Parpadea para verificar..."
                color = (255, 255, 255)  # Blanco o gris claro

        # Almacenar resultado para dibujar después
        resultados.append((x, y, w, h, texto, color))

    # ------ 4.5 Limpiar usuarios_confirmados caducados ------
    usuarios_confirmados = [
        u for u in usuarios_confirmados
        if (tiempo_actual - u["timestamp"] < tiempo_validez)
    ]

    # ------ 4.6 Dibujar rectángulos y textos en el frame ------
    for (x, y, w, h, texto, color) in resultados:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            texto,
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Reconocimiento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# 5. Liberar recursos
# =========================

cap.release()
cv2.destroyAllWindows()

import os
import cv2
import numpy as np
import dlib
import time
import json
import sys
from imutils import face_utils


# Reconocimiento LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Models/modeloLBPHFace.xml')

# Detector DNN de OpenCV
dnn_proto = 'Models/deploy.prototxt'
dnn_model = 'Models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
if not os.path.isfile(dnn_proto) or not os.path.isfile(dnn_model):
    print(f"Error: no se encuentra el modelo DNN en:\n  {dnn_proto}\n  {dnn_model}")
    sys.exit(1)
net_dnn = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
conf_threshold = 0.5  # Umbral mínimo para aceptar detección DNN

# Predictor Dlib
predictor_dlib = dlib.shape_predictor('Models/shape_predictor_68_face_landmarks.dat')

# Cargar lista de nombres desde mapping JSON
with open('Models/mapping_labels.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)
label2name = {int(v): k for k, v in mapping.items()}

# Umbral de confianza LBPH
umbral_reconocimiento = 60

# Tiempo maximo para mantener usuario confirmado
tiempo_validez = 5.0  # segundos

# Tiempo para superar prueba liveness (parpadeo)
life_timeout = 5.0  # segundos

usuarios_confirmados = []

active_faces = {}


def detectar_parpadeo(shape_np):
    """
    Recibe array (68,2) de landmarks.
    Calcula EAR y devuelve (ear, True si EAR < 0.18).
    """
    ojo_izq = shape_np[36:42]
    ojo_der = shape_np[42:48]
    ratio_izq = np.linalg.norm(ojo_izq[1] - ojo_izq[5]) / np.linalg.norm(ojo_izq[0] - ojo_izq[3])
    ratio_der = np.linalg.norm(ojo_der[1] - ojo_der[5]) / np.linalg.norm(ojo_der[0] - ojo_der[3])
    ear = (ratio_izq + ratio_der) / 2.0
    return ear, (ear < 0.18)

def encontrar_usuario_existente(cx, cy, tiempo_actual):
    """
    Comprueba si ya existe un usuario confirmado cerca de (cx, cy) y no caducado.
    Devuelve la entrada si la encuentra, sino None.
    """
    for usuario in usuarios_confirmados:
        ux, uy = usuario["pos"]
        dist = np.hypot(ux - cx, uy - cy)
        if dist < 50 and (tiempo_actual - usuario["timestamp"] < tiempo_validez):
            return usuario
    return None


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo abrir la camara.")
    sys.exit(1)

cv2.namedWindow("Reconocimiento Facial", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()
    alto, ancho = frame.shape[:2]
    tiempo_actual = time.time()

    # Limpiar active_faces expirados
    to_delete = []
    for face_key, data in active_faces.items():
        if tiempo_actual - data["start_time"] > (life_timeout + 1.0):
            to_delete.append(face_key)
    for k in to_delete:
        del active_faces[k]

    # Detectar rostros con DNN
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net_dnn.setInput(blob)
    detections = net_dnn.forward()

    faces = []
    for i in range(detections.shape[2]):
        confianza = float(detections[0, 0, i, 2])
        if confianza < conf_threshold:
            continue
        x1 = int(detections[0, 0, i, 3] * ancho)
        y1 = int(detections[0, 0, i, 4] * alto)
        x2 = int(detections[0, 0, i, 5] * ancho)
        y2 = int(detections[0, 0, i, 6] * alto)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(ancho - 1, x2); y2 = min(alto - 1, y2)
        w = x2 - x1; h = y2 - y1
        if w < 50 or h < 50:
            continue
        faces.append((x1, y1, w, h))

    resultados = []

    # Procesar cada cara detectada
    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2

        x1, y1 = x, y
        x2, y2 = x + w, y + h
        rect_asociado = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

        ear = None
        detected_blink = False
        texto_ear = None

        try:
            shape = predictor_dlib(gray, rect_asociado)
            shape_np = face_utils.shape_to_np(shape)

            for (px, py) in shape_np:
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                
            # Calcular EAR y determinar parpadeo
            ear, detected_blink = detectar_parpadeo(shape_np)
            texto_ear = f"EAR: {ear:.2f}"
            
        except Exception:
            # Si no podemos extraer landmarks, dejamos todo en None
            ear = None
            texto_ear = None
            detected_blink = False

        # Clave redondeada para identificar la cara en active_faces
        fx = int(round(cx / 50.0) * 50)
        fy = int(round(cy / 50.0) * 50)
        face_key = (fx, fy)

        if face_key not in active_faces:
            active_faces[face_key] = {"start_time": tiempo_actual, "blinked": False}
        state = active_faces[face_key]
        elapsed = tiempo_actual - state["start_time"]

        # Si detectamos parpadeo ahora, marcamos estado como blinked
        if detected_blink:
            state["blinked"] = True

        
        if not state["blinked"] and elapsed < life_timeout:
            # Pedir al usuario que parpadee
            if ear is not None:
                texto = f"Parpadea para verificar... | EAR: {ear:.2f}"
            else:
                texto = "Parpadea para verificar..."
            color = (255, 255, 255)


        elif state["blinked"]:
            roi_color = aux_frame[y1:y2, x1:x2]
            rostro_gray = cv2.cvtColor(cv2.resize(roi_color, (150, 150)), cv2.COLOR_BGR2GRAY)
            id_usuario, confianza = face_recognizer.predict(rostro_gray)

            if confianza < umbral_reconocimiento and id_usuario in label2name:
                nombre = label2name[id_usuario]
                texto = f"{nombre} - {confianza:.2f}"
                color = (0, 255, 0)
                
                # Actualizar o añadir a usuarios_confirmados
                usuario_existente = encontrar_usuario_existente(cx, cy, tiempo_actual)
                if not usuario_existente:
                    usuarios_confirmados.append({
                        "pos": (cx, cy),
                        "label": nombre,
                        "timestamp": tiempo_actual
                    })
                    
                else:
                    usuario_existente["timestamp"] = tiempo_actual
                    
            else:
                texto = f"Desconocido | EAR: {ear:.2f}" if ear is not None else "Desconocido"
                color = (0, 0, 255)

        # No parpadeo y se agoto life_timeout => foto estatica
        else:
            if ear is not None:
                texto = f"Imagen detectada | EAR: {ear:.2f}"
            else:
                texto = "Imagen detectada"
            color = (0, 128, 255)

        resultados.append((x1, y1, w, h, texto, color, texto_ear))

    # Limpiar usuarios_confirmados caducados
    usuarios_confirmados = [
        u for u in usuarios_confirmados
        if (tiempo_actual - u["timestamp"] < tiempo_validez)
    ]

    # Dibujar todos los resultados en el frame
    for (x, y, w, h, texto, color, texto_ear) in resultados:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Si existe texto_ear, dibujarlo arriba del recuadro
        if texto_ear is not None:
            y_arriba = max(y - 10, 20)
            cv2.putText(frame,
                        texto_ear,
                        (x, y_arriba),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2)

        # Texto principal debajo del recuadro
        cv2.putText(frame,
                    texto,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    cv2.imshow("Reconocimiento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

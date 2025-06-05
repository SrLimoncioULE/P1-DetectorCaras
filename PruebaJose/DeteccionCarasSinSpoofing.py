import os
import cv2
import numpy as np
import time
import json


# Reconocimiento LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Models/modeloLBPHFace.xml')

# Detector DNN de OpenCV
dnn_proto = 'Models/deploy.prototxt'
dnn_model = 'Models/res10_300x300_ssd_iter_140000_fp16.caffemodel'

if not os.path.isfile(dnn_proto) or not os.path.isfile(dnn_model):
    print(f"Error: no se encuentra el modelo DNN en:\n  {dnn_proto}\n  {dnn_model}")
    exit(1)

net_dnn = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
conf_threshold = 0.5

# Cargar nombres desde mapping JSON
with open('Models/mapping_labels.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)
label2name = {int(v): k for k, v in mapping.items()}

# Umbral de confianza LBPH
umbral_reconocimiento = 60

# Tiempo max que un usuario se mantiene confirmado
tiempo_validez = 3.0

usuarios_confirmados = []


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
    exit(1)

cv2.namedWindow("Reconocimiento Facial", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()
    tiempo_actual = time.time()

    # Detectar rostros con DNN
    alto, ancho = frame.shape[:2]
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
        
        # Recortar y redimensionar para LBPH
        roi_color = aux_frame[y:y + h, x:x + w]
        rostro_gray = cv2.cvtColor(cv2.resize(roi_color, (150, 150)), cv2.COLOR_BGR2GRAY)

        # Reconocer con LBPH
        id_usuario, confianza = face_recognizer.predict(rostro_gray)
        if confianza < umbral_reconocimiento and id_usuario in label2name:
            nombre = label2name[id_usuario]
            texto = f"{nombre} - {confianza:.2f}"
            color = (0, 255, 0)

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
            # Si NO reconocido, marcamos desconocido
            texto = "Desconocido"
            color = (0, 0, 255)

        resultados.append((x, y, w, h, texto, color))

    # Limpiar usuarios_confirmados caducados
    usuarios_confirmados = [
        u for u in usuarios_confirmados
        if (tiempo_actual - u["timestamp"] < tiempo_validez)
    ]

    # Dibujar resultados en el frame
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

cap.release()
cv2.destroyAllWindows()

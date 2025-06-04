import os
import cv2
import numpy as np
import time
import json

# =========================
# 1. Inicializar modelos y parámetros
# =========================

# 1.1. Reconocimiento LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Models/modeloLBPHFace.xml')

# 1.2. Detector DNN de OpenCV (Res10 SSD)
dnn_proto = 'Models/deploy.prototxt'
dnn_model = 'Models/res10_300x300_ssd_iter_140000_fp16.caffemodel'

if not os.path.isfile(dnn_proto) or not os.path.isfile(dnn_model):
    print(f"Error: no se encuentra el modelo DNN en:\n  {dnn_proto}\n  {dnn_model}")
    exit(1)

net_dnn = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
conf_threshold = 0.5  # Umbral mínimo para aceptar detección DNN

# 1.3. Cargar lista de nombres desde mapping JSON en lugar de hardcode
with open('Models/mapping_labels.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)  # ej. {"Jose": 0, "Marcelo": 1, ...}
label2name = {int(v): k for k, v in mapping.items()}

# 1.4. Umbral de confianza LBPH: solo se considera “reconocido” si confianza < umbral
umbral_reconocimiento = 60

# 1.5. Tiempo máximo que mantenemos a un usuario “confirmado” (en segundos)
tiempo_validez = 3.0

# =========================
# 2. Variables de estado global
# =========================

# 2.1. Para almacenar usuarios reconocidos recientemente
usuarios_confirmados = []
# Cada entrada: {"pos": (cx, cy), "label": str, "timestamp": float}

# =========================
# 3. Funciones auxiliares
# =========================

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

# =========================
# 4. Bucle principal
# =========================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
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

    # 4.1 Detectar rostros con DNN
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

    # 4.2 Procesar cada cara detectada
    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        # Recortar y redimensionar para LBPH
        roi_color = aux_frame[y:y + h, x:x + w]
        rostro_gray = cv2.cvtColor(cv2.resize(roi_color, (150, 150)), cv2.COLOR_BGR2GRAY)

        # 4.2.1 Intentar reconocer con LBPH
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
            # Si NO reconocido, marcamos directamente como desconocido
            texto = "Desconocido"
            color = (0, 0, 255)

        resultados.append((x, y, w, h, texto, color))

    # 4.3 Limpiar usuarios_confirmados caducados
    usuarios_confirmados = [
        u for u in usuarios_confirmados
        if (tiempo_actual - u["timestamp"] < tiempo_validez)
    ]

    # 4.4 Dibujar resultados en el frame
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

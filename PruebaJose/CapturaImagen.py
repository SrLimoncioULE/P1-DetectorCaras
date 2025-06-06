import cv2
import os
import time
import numpy as np


# Configuracion inicial

# Numero max de imagenes a guardar
n_max = 100

# Tamaño del resize
FACE_W, FACE_H = 150, 150

# Rutas del detector DNN
DNN_PROTO = os.path.join("Models", "deploy.prototxt")
DNN_MODEL = os.path.join("Models", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

if not os.path.isfile(DNN_PROTO) or not os.path.isfile(DNN_MODEL):
    print("Error: no se encuentra el modelo DNN en:")
    print(f"  {DNN_PROTO}")
    print(f"  {DNN_MODEL}")
    print("Colocalos e intente de nuevo/")
    exit(1)

# Cargamos el detector DNN
net_dnn = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
CONF_THRESHOLD = 0.5  # Umbral mínimo para aceptar detección

# Carpeta donde se guardan los rostros
BASE_DIR = "Rostros"

# Pedimos nombre de usuario
nombre = input("\nNombre: ").strip()

# Crear carpetas si no existen
if not os.path.isdir(BASE_DIR):
    os.makedirs(BASE_DIR)
    print(f"Carpeta creada: {BASE_DIR}")

user_folder = os.path.join(BASE_DIR, nombre)
if not os.path.isdir(user_folder):
    os.makedirs(user_folder)
    print(f"Carpeta creada: {user_folder}")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo abrir la camara.")
    exit(1)
  
cv2.namedWindow("Captura Imagenes", cv2.WINDOW_NORMAL)

count = 0
print(f"Iniciando captura de {n_max} imágenes para '{nombre}'...")

while count < n_max:
    # Extraemos el frame
    ret, frame = cap.read()
    if not ret:
        print("Error: no se pudo leer el frame de la camara.")
        break

    # Modo espejo
    frame = cv2.flip(frame, 1)
    aux = frame.copy()
    alto, ancho = frame.shape[:2]

    # Convertir a blob 300×300 para el DNN
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )
    net_dnn.setInput(blob)
    detections = net_dnn.forward()

    rostro_detectado = False

    # Recorremos todas las detecciones y tomamos la primera valida
    for i in range(detections.shape[2]):
        confianza = float(detections[0, 0, i, 2])
        if confianza < CONF_THRESHOLD:
            continue

        # Coordenadas absolutas
        x1 = int(detections[0, 0, i, 3] * ancho)
        y1 = int(detections[0, 0, i, 4] * alto)
        x2 = int(detections[0, 0, i, 5] * ancho)
        y2 = int(detections[0, 0, i, 6] * alto)

        # Limites dentro de la imagen
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(ancho - 1, x2)
        y2 = min(alto - 1, y2)

        w = x2 - x1
        h = y2 - y1

        # Descartar detecciones demasiado pequeñas
        if w < 50 or h < 50:
            continue

        # Extraer ROI en color y convertir a gris + ecualizar
        roi_color = aux[y1:y1 + h, x1:x1 + w]
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.equalizeHist(roi_gray)

        # Redimensionar a 150×150
        rostro_resized = cv2.resize(roi_gray, (FACE_W, FACE_H), interpolation=cv2.INTER_CUBIC)

        # Guardar el archivo
        filename = f"{nombre}_{count}.jpg"
        filepath = os.path.join(user_folder, filename)
        cv2.imwrite(filepath, rostro_resized)

        count += 1
        print(f"[{count}/{n_max}] Imagen guardada: {filepath}")

        # Dibujar rectangulo y etiqueta de contador
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{count}/{n_max}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        rostro_detectado = True
        break

    # Si no detecto ningun rostro avisamos usuario
    if not rostro_detectado:
        cv2.putText(
            frame,
            "No se detecta cara. Ajusta tu posicion.",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # Mostrar contador general en pantalla
    cv2.putText(
        frame,
        f"Captura {count}/{n_max}",
        (50, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Captura DNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Si ya capturamos un rostro, pausamos 0.2 s para no duplicar imagenes
    if rostro_detectado:
        time.sleep(0.2)


cap.release()
cv2.destroyAllWindows()
print(f"Proceso completado. Se han guardado {count} rostros en la carpeta '{user_folder}'.")

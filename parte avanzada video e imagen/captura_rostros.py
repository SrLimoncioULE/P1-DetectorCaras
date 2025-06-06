
import cv2
import os

# ---------------- CONFIGURACIÓN ---------------- #
nombre_persona = input("Nombre de la persona: ").strip()
entrada = input("Número de imágenes (por defecto 250): ").strip()

num_imagenes = 250
if entrada.isdigit():
    num_imagenes = int(entrada)

dataPath = 'Data'
personPath = os.path.join(dataPath, nombre_persona)

# Crear carpeta si no existe
if not os.path.exists(personPath):
    os.makedirs(personPath)
    print(f"[INFO] Carpeta creada: {personPath}")

# Cargar clasificadores Haar de forma robusta
face_frontal_path = r'C:\Users\Usuario\.spyder-py3\extraciondeRostro\haarcascade_frontalface_default.xml'
face_profile_path = r'C:\Users\Usuario\.spyder-py3\extraciondeRostro\haarcascade_profileface.xml'
eye_cascade_path = r'C:\Users\Usuario\.spyder-py3\extraciondeRostro\haarcascade_eye.xml'

for path in [face_frontal_path, face_profile_path, eye_cascade_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Clasificador no encontrado: {path}")

face_frontal = cv2.CascadeClassifier(face_frontal_path)
face_profile = cv2.CascadeClassifier(face_profile_path)
eyeClassif = cv2.CascadeClassifier(eye_cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("[ERROR] No se pudo abrir la cámara.")

count = 0
print("[INFO] Iniciando captura de rostros...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el frame de la cámara.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = face_frontal.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        faces = face_profile.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro_gray = gray[y:y + h, x:x + w]

        ojos = eyeClassif.detectMultiScale(rostro_gray)

        if len(ojos) >= 1:
            rostro_resized = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            file_path = os.path.join(personPath, f'rostro_{count:03d}.jpg')
            cv2.imwrite(file_path, rostro_resized)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 255), 2)
            cv2.imshow('Rostro', rostro_resized)

            if count >= num_imagenes:
                break

    texto = f'Capturadas: {count}/{num_imagenes}'
    cv2.putText(frame, texto, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == 27 or count >= num_imagenes:  # ESC o límite alcanzado
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Captura finalizada. Total imágenes capturadas: {count}")

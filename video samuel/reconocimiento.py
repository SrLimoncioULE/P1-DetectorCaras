import cv2
import dlib
import os
import numpy as np
from imutils import face_utils

#  RUTAS 
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'modelos', 'modeloLBPHFace.xml')
people_path = os.path.join(base_dir, 'peopleList.npy')
shape_predictor_path = r'C:\Users\Usuario\.spyder-py3\extraciondeRostro\shape_predictor_68_face_landmarks.dat'

#  VALIDACIÓN DE ARCHIVOS 
if not os.path.exists(model_path):
    raise FileNotFoundError(f"[ERROR] Modelo no encontrado: {model_path}")
if not os.path.exists(people_path):
    raise FileNotFoundError(f"[ERROR] Etiquetas no encontradas: {people_path}")
if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"[ERROR] Modelo dlib no encontrado: {shape_predictor_path}")

#  CARGA DE MODELOS 
modelo = cv2.face.LBPHFaceRecognizer_create()
modelo.read(model_path)
peopleList = np.load(people_path, allow_pickle=True)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

face_frontal = cv2.CascadeClassifier(r'C:\Users\Usuario\.spyder-py3\extraciondeRostro\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\Usuario\.spyder-py3\extraciondeRostro\haarcascade_eye.xml')

if face_frontal.empty() or eye_cascade.empty():
    raise FileNotFoundError("[ERROR] Clasificadores Haar no se cargaron correctamente.")

#  VARIABLES DE ANTI-SPOOFING 
last_eye_positions = []
last_face_position = None
frames_static_eyes = 0
frames_static_face = 0
threshold_static = 60

cap = cv2.VideoCapture(0)
print('[INFO] Iniciando reconocimiento facial...')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_frontal.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro_gray = gray[y:y + h, x:x + w]
        rostro_color = frame[y:y + h, x:x + w]

        rostro_gray = cv2.equalizeHist(rostro_gray)
        rostro_gray = cv2.GaussianBlur(rostro_gray, (3, 3), 0)
        rostro_resized = cv2.resize(rostro_gray, (150, 150), interpolation=cv2.INTER_CUBIC)

        label_id, confidence = modelo.predict(rostro_resized)
        ojos = eye_cascade.detectMultiScale(rostro_gray)
        ojos_data = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in ojos]

        #  Anti-Spoofing básico (movimiento de ojos/rostro) =
        spoof_static = False
        spoof_video = False

        if len(ojos_data) >= 2:
            if last_eye_positions == ojos_data:
                frames_static_eyes += 1
            else:
                frames_static_eyes = 0
            last_eye_positions = ojos_data
            if frames_static_eyes > threshold_static:
                spoof_static = True
        else:
            spoof_static = True

        current_face_position = (x, y, w, h)
        if last_face_position == current_face_position:
            frames_static_face += 1
        else:
            frames_static_face = 0
        last_face_position = current_face_position
        if frames_static_face > threshold_static:
            spoof_video = True

        #  Detección de bordes — posible suplantación con imagen/foto 
        spoof_borders = False
        edges = cv2.Canny(rostro_gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                aspect_ratio = w_c / float(h_c)
                if 0.8 < aspect_ratio < 1.2:
                    spoof_borders = True
                    break

        # Clasificación de resultado 
        if spoof_static or spoof_video or spoof_borders:
            texto = "Spoofing detectado"
            color = (0, 0, 255)
        elif confidence < 50:
            label = peopleList[label_id]
            texto = f'{label} (Alta confianza - {round(confidence, 2)})'
            color = (0, 255, 0)
        elif confidence < 80:
            label = peopleList[label_id]
            texto = f'{label} (Media confianza - {round(confidence, 2)})'
            color = (0, 255, 255)
        else:
            texto = 'Desconocido'
            color = (0, 0, 255)

        #  Dibujar resultados 
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Indicador visual de precisión
        conf_display = int(max(0, 100 - confidence))
        cv2.rectangle(frame, (x, y + h + 5), (x + conf_display, y + h + 20), color, -1)
        cv2.putText(frame, f'{conf_display}%', (x, y + h + 35), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        for (ex, ey, ew, eh) in ojos:
            cv2.rectangle(rostro_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

    #  Detección de puntos faciales (Dlib) 
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        for (px, py) in shape_np:
            cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

    cv2.imshow('Reconocimiento Facial + Anti-Spoofing', frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print('[INFO] Reconocimiento finalizado.')

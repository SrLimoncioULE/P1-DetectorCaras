import cv2
import os
import time  # Importamos time para agregar pausas

# Número máximo de imágenes a guardar
n_max = 200

# Listas para almacenar etiquetas e imágenes de rostros
labels = []
facesData = []
label = 0  # Etiqueta inicial

# Solicitar nombre del usuario
nombre = input("Introduce tu nombre: ")

# Crear carpeta para almacenar los rostros si no existe
if not os.path.exists('Rostros'):
    print('Carpeta creada: Rostros')
    os.makedirs('Rostros')

# Crear carpeta específica para cada usuario dentro de 'Rostros'
user_folder = f'Rostros/{nombre}'
if not os.path.exists(user_folder):
    print(f'Carpeta creada: {user_folder}')
    os.makedirs(user_folder)

# Inicializar captura de video
cap = cv2.VideoCapture(0)
face_frontal = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
face_alt = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_alt.xml')
face_perfil = cv2.CascadeClassifier('Cascades/haarcascade_profileface.xml')
count = 0

# Crear una única ventana
cv2.namedWindow('Detección de rostros', cv2.WINDOW_NORMAL)

while count < n_max:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Voltear la imagen para simular efecto espejo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    # Detectar rostros con múltiples clasificadores
    faces_frontal = face_frontal.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(40, 40))
    faces_alt = face_alt.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(40, 40))
    faces_perfil_izq = face_perfil.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    flipped_gray = cv2.flip(gray, 1)  
    faces_perfil_der = face_perfil.detectMultiScale(flipped_gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    # Procesar cada detección con pausa
    for face_list, color in [(faces_frontal, (0, 255, 0)), (faces_alt, (0, 128, 255)), 
                             (faces_perfil_izq, (255, 0, 0)), (faces_perfil_der, (255, 255, 0))]:
        for (x, y, w, h) in face_list:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Guardar la imagen dentro de la carpeta del usuario
            fileName = f'{user_folder}/{nombre}_{count}.jpg'
            cv2.imwrite(fileName, rostro)

            labels.append(0)
            facesData.append(cv2.imread(fileName, 0))

            count += 1
            print(f'Rostro guardado: {count}/{n_max}')

            time.sleep(0.1)  #Pausa entre capturas

    # Mostrar contador en pantalla
    cv2.putText(frame, f'Captura {count}/{n_max}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detección de rostros', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f'Proceso completado. Se han guardado {count} rostros en la carpeta {user_folder}.')
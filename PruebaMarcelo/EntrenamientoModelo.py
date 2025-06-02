import cv2
import os
import numpy as np

# Ruta donde están almacenadas las carpetas de usuarios
dataPath = "Rostros"  
userFolders = os.listdir(dataPath)  # Lista de carpetas de usuarios
print(f"Usuarios encontrados: {userFolders}")

# Listas para almacenar los datos y etiquetas
facesData = []
labels = []
label = 0  # Etiqueta inicial

# Recorrer cada carpeta de usuario
for user in userFolders:
    userPath = os.path.join(dataPath, user)  
    
    if os.path.isdir(userPath):  # Verificar que sea un directorio
        imageFiles = os.listdir(userPath)  # Lista de imágenes en la carpeta del usuario
        
        print(f"Cargando imágenes de: {user} (Etiqueta: {label})")

        for fileName in imageFiles:
            filePath = os.path.join(userPath, fileName)
            
            # Verificar que el archivo sea una imagen
            if fileName.endswith(".jpg") or fileName.endswith(".png"):
                image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises

                facesData.append(image)
                labels.append(label)  # Asignar etiqueta única a cada usuario

        label += 1  # Incrementar etiqueta para el siguiente usuario

print(f"Total de imágenes cargadas: {len(facesData)}")

# Crear el modelo LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenar el modelo con las imágenes y etiquetas
print("Entrenando modelo LBPH...")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado para pruebas futuras
face_recognizer.write('Models/modeloLBPHFace.xml')
print("Modelo LBPH almacenado correctamente.")
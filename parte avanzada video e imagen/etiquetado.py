import cv2
import os
import numpy as np

#  CONFIGURACIÓN  #
base_dir = os.path.dirname(os.path.abspath(__file__))  # Carpeta actual
dataPath = os.path.join(base_dir, '..', 'Data')  # Carpeta Data un nivel arriba

# Validar existencia de Data
if not os.path.exists(dataPath):
    raise FileNotFoundError(f"[ERROR] No se encuentra la carpeta: {dataPath}")

# Obtener solo las carpetas (nombres de personas)
peopleList = [name for name in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, name))]

labels = []
facesData = []
label = 0

print('[INFO] Leyendo las imágenes y etiquetas...')

for personName in peopleList:
    personPath = os.path.join(dataPath, personName)
    for fileName in os.listdir(personPath):
        if not fileName.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue  # Ignorar archivos no imagen

        imagePath = os.path.join(personPath, fileName)
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)  # Leer directamente en escala de grises
        if img is None:
            print(f"[WARNING] No se pudo leer la imagen: {imagePath}")
            continue

        # Asegurar tamaño correcto
        if img.shape != (150, 150):
            img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)

        facesData.append(img)
        labels.append(label)

    label += 1

print(f'[INFO] Total de imágenes leídas: {len(facesData)}')
print(f'[INFO] Total de etiquetas únicas: {len(set(labels))}')

# Convertir a arrays numpy para entrenar
facesData = np.array(facesData)
labels = np.array(labels)

# Guardar arrays en disco para entrenamiento
np.save(os.path.join(base_dir, 'facesData.npy'), facesData)
np.save(os.path.join(base_dir, 'labels.npy'), labels)
np.save(os.path.join(base_dir, 'peopleList.npy'), peopleList)

print('[INFO] Arrays guardados con éxito.')

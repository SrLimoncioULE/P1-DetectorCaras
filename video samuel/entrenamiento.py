import cv2
import numpy as np
import os

#  CONFIGURACIÓN  #
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'modelos')

# Crear carpeta modelos si no existe
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Cargar datos guardados
try:
    facesData = np.load(os.path.join(base_dir, 'facesData.npy'), allow_pickle=True)
    labels = np.load(os.path.join(base_dir, 'labels.npy'))
except FileNotFoundError as e:
    print(f'[ERROR] Archivo no encontrado: {e}')
    exit(1)

if len(facesData) == 0 or len(labels) == 0:
    print('[ERROR] No hay datos para entrenar.')
    exit(1)

print('[INFO] Procesando imágenes para entrenamiento...')

processed_faces = []
for face in facesData:
    # Asegurar que la imagen esté en escala de grises y tamaño 150x150
    if face.ndim == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
    processed_faces.append(face_resized)

processed_faces = np.array(processed_faces)

print('[INFO] Iniciando entrenamiento con LBPHFaceRecognizer...')

# Crear y entrenar el recognizer LBPH
try:
    lbph = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print('[ERROR] No se encontró el método LBPHFaceRecognizer_create. Asegúrate de tener instalada la versión correcta de OpenCV con contrib.')
    exit(1)

lbph.train(processed_faces, np.array(labels))

# Guardar modelo entrenado
modelo_path = os.path.join(model_dir, 'modeloLBPHFace.xml')
lbph.write(modelo_path)

print(f'[INFO] Modelo LBPH guardado en: {modelo_path}')

import cv2
import os
import numpy as np
import json
import sys

# =======================================
# 0. Configuración y rutas
# =======================================
DATA_PATH = "Rostros"                      # Carpeta con subcarpetas por usuario
MODELS_DIR = "Models"                      # Carpeta donde se guardarán modelo y mapping
MODEL_FILENAME = "modeloLBPHFace.xml"
MAPPING_FILENAME = "mapping_labels.json"
FACE_W, FACE_H = 150, 150                  # Tamaño usado en la captura

# =======================================
# 1. Verificaciones iniciales
# =======================================
if not os.path.isdir(DATA_PATH):
    print(f"Error: no existe la carpeta de datos '{DATA_PATH}'.\n"
          "  → Asegúrate de ejecutar primero el script de captura para crear las subcarpetas con imágenes.")
    sys.exit(1)

if not os.path.isdir(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Carpeta creada para guardar modelos: {MODELS_DIR}")

# =======================================
# 2. Listar carpetas de usuarios y mapping
# =======================================
user_folders = [
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
]
if len(user_folders) == 0:
    print(f"Error: no se encontró ninguna subcarpeta dentro de '{DATA_PATH}'.")
    sys.exit(1)

user_folders.sort()  # Orden alfabético
mapping = { user_folders[i]: i for i in range(len(user_folders)) }

# Guardar mapping en JSON
mapping_path = os.path.join(MODELS_DIR, MAPPING_FILENAME)
with open(mapping_path, "w", encoding="utf-8") as fp:
    json.dump(mapping, fp, ensure_ascii=False, indent=4)
print(f"Mapping etiquetas guardado en: {mapping_path}\n")
print("Usuario → Etiqueta:")
for user, lbl in mapping.items():
    print(f"  - {user:15s} → {lbl}")

# =======================================
# 3. Cargar imágenes y etiquetas
# =======================================
faces_data = []      # Lista de imágenes en escala de grises (150×150)
labels = []          # Etiqueta entera asociada a cada imagen
count_per_user = { user: 0 for user in user_folders }

for user_name, lbl in mapping.items():
    folder_path = os.path.join(DATA_PATH, user_name)
    print(f"\nProcesando carpeta de usuario '{user_name}' (Etiqueta: {lbl})")

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ⚠️ No se pudo leer '{img_path}', omitiendo.")
            continue

        # Verificar tamaño; debería ser 150×150 según captura
        h, w = img.shape[:2]
        if (w, h) != (FACE_W, FACE_H):
            try:
                img = cv2.resize(img, (FACE_W, FACE_H), interpolation=cv2.INTER_CUBIC)
            except Exception:
                print(f"  ⚠️ No se pudo redimensionar '{img_path}', omitiendo.")
                continue

        # Ecualizar histograma
        img = cv2.equalizeHist(img)

        faces_data.append(img)
        labels.append(lbl)
        count_per_user[user_name] += 1

    print(f"  Imágenes válidas cargadas para '{user_name}': {count_per_user[user_name]}")

total_images = len(faces_data)
print(f"\nTotal de imágenes procesadas para entrenamiento: {total_images}")

if total_images < 10:
    print("⚠️ Advertencia: dispones de menos de 10 imágenes en total.")
    print("   Se recomienda capturar al menos 20–30 fotos por usuario para un LBPH estable.")

# =======================================
# 4. Entrenar modelo LBPH
# =======================================
print("\nIniciando entrenamiento del modelo LBPH...")
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces_data, np.array(labels))
    print("  ✔️ Entrenamiento completado.")
except Exception as e:
    print(f"Error durante el entrenamiento LBPH: {e}")
    sys.exit(1)

# =======================================
# 5. Guardar el modelo entrenado
# =======================================
model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
face_recognizer.write(model_path)
print(f"Modelo LBPH guardado en: {model_path}\n")
print("✅ Proceso finalizado. El modelo está listo para reconocimiento.")

import cv2
import os
import numpy as np
import json
import sys


DATA_PATH = "Rostros"
MODELS_DIR = "Models"
MODEL_FILENAME = "modeloLBPHFace.xml"
MAPPING_FILENAME = "mapping_labels.json"
FACE_W, FACE_H = 150, 150


if not os.path.isdir(DATA_PATH):
    print(f"Error: no existe la carpeta de datos '{DATA_PATH}'.\n")
    sys.exit(1)

if not os.path.isdir(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Carpeta creada para guardar modelos: {MODELS_DIR}")


user_folders = [
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
]
if len(user_folders) == 0:
    print(f"Error: no se encontro ninguna subcarpeta dentro de '{DATA_PATH}'.")
    sys.exit(1)

user_folders.sort()  # Orden alfabetico
mapping = { user_folders[i]: i for i in range(len(user_folders)) }

# Guardar mapping en JSON
mapping_path = os.path.join(MODELS_DIR, MAPPING_FILENAME)
with open(mapping_path, "w", encoding="utf-8") as fp:
    json.dump(mapping, fp, ensure_ascii=False, indent=4)
print(f"Mapping etiquetas guardado en: {mapping_path}\n")
print("Usuario → Etiqueta:")
for user, lbl in mapping.items():
    print(f"  - {user:15s} → {lbl}")


faces_data = []
labels = []
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
            print(f"  No se pudo leer '{img_path}', omitiendo.")
            continue

        # Verificar tamaño 150×150
        h, w = img.shape[:2]
        if (w, h) != (FACE_W, FACE_H):
            try:
                img = cv2.resize(img, (FACE_W, FACE_H), interpolation=cv2.INTER_CUBIC)
            except Exception:
                print(f"  No se pudo redimensionar '{img_path}', omitiendo.")
                continue

        # Ecualizar histograma
        img = cv2.equalizeHist(img)

        faces_data.append(img)
        labels.append(lbl)
        count_per_user[user_name] += 1

    print(f"  Imagenes cargadas para '{user_name}': {count_per_user[user_name]}")

total_images = len(faces_data)
print(f"\nTotal de imagenes procesadas: {total_images}")


print("\nIniciando entrenamiento del modelo LBPH...")
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces_data, np.array(labels))
    print("  Entrenamiento completado.")
except Exception as e:
    print(f"Error durante el entrenamiento LBPH: {e}")
    sys.exit(1)


model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
face_recognizer.write(model_path)
print(f"Modelo LBPH guardado en: {model_path}\n")
print("Proceso finalizado. El modelo esta listo para reconocimiento.")

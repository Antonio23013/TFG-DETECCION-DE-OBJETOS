import os
import glob
import cv2
import random
import albumentations as A

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
# Rutas relativas a la carpeta de entrenamiento
DATASET_DIR = os.path.abspath('./data/YOLO/train')
IMG_DIR = os.path.join(DATASET_DIR, 'images')
LBL_DIR = os.path.join(DATASET_DIR, 'labels')

# El objetivo de cajas delimitadoras (Bounding Boxes) para balancear las clases
TARGET_BOXES = 17814  

# ==============================================================================
# 🎨 PIPELINE DE TRANSFORMACIÓN FOTOMÉTRICA (ALBUMENTATIONS)
# ==============================================================================
# Prohibido alterar geometría. Solo simulamos condiciones atmosféricas y de sensor.
color_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.6),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),  # Simula ruido del sensor orbital
    A.Blur(blur_limit=3, p=0.2)                   # Simula desenfoque de movimiento
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==============================================================================
# 📊 FUNCIONES AUXILIARES
# ==============================================================================
def count_current_labels():
    """Lee todos los txt actuales y cuenta cuántas instancias hay de cada clase."""
    counts = {0: 0, 1: 0, 2: 0}
    label_files = glob.glob(os.path.join(LBL_DIR, '*.txt'))
    
    for file in label_files:
        if file.endswith('classes.txt'): continue
        with open(file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if parts:
                    cls = int(float(parts[0]))
                    if cls in counts:
                        counts[cls] += 1
    return counts

# ==============================================================================
# 🚀 EJECUCIÓN DEL AUMENTO DE DATOS
# ==============================================================================
def augment_photometric():
    print(f"📂 Escaneando directorio: {IMG_DIR}")
    base_images = [f for f in glob.glob(os.path.join(IMG_DIR, '*.*')) if f.endswith(('.jpg', '.png'))]
    
    current_counts = count_current_labels()
    print("\n📊 Conteo actual de clases antes de aumentar:")
    print(f"   Civil (0): {current_counts[0]} | Heli (1): {current_counts[1]} | Militar (2): {current_counts[2]}")
    print(f"🎯 Objetivo por clase: {TARGET_BOXES}\n")

    images_generated = 0

    print("🎨 Iniciando Data Augmentation Fotométrico...")
    # Bucle hasta que las clases minoritarias (0 y 1) alcancen el target
    while current_counts[0] < TARGET_BOXES or current_counts[1] < TARGET_BOXES:
        img_path = random.choice(base_images)
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        label_path = os.path.join(LBL_DIR, f"{name_no_ext}.txt")

        if not os.path.exists(label_path):
            continue

        bboxes, class_labels = [], []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    class_labels.append(int(float(parts[0])))

        # Solo aplicamos si la imagen contiene clases que aún no han llegado al objetivo
        needs_augmentation = any(c in [0, 1] and current_counts[c] < TARGET_BOXES for c in class_labels)
        if not needs_augmentation or not bboxes:
            continue

        # Leer imagen
        image = cv2.imread(img_path)
        if image is None: 
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Aplicar transformación
        try:
            transformed = color_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            t_image = transformed['image']
            t_bboxes = transformed['bboxes']
            t_labels = transformed['class_labels']
        except Exception:
            continue # Si falla (caja fuera de límites, etc), saltamos a otra

        if not t_bboxes: 
            continue

        # Actualizar contadores
        for c in t_labels:
            if c in current_counts:
                current_counts[c] += 1
        
        images_generated += 1

        # Guardar nueva imagen
        new_name = f"{name_no_ext}_aug_color_{images_generated}"
        cv2.imwrite(os.path.join(IMG_DIR, f"{new_name}.jpg"), cv2.cvtColor(t_image, cv2.COLOR_RGB2BGR))

        # Guardar nuevas etiquetas
        with open(os.path.join(LBL_DIR, f"{new_name}.txt"), 'w') as f:
            for bbox, label in zip(t_bboxes, t_labels):
                f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        if images_generated % 500 == 0:
            print(f" -> Generadas {images_generated} imágenes nuevas. (Civil: {current_counts[0]}, Heli: {current_counts[1]})")

    print(f"\n✅ COMPLETADO. Se han generado {images_generated} imágenes sintéticas por variación de color.")

if __name__ == "__main__":
    augment_photometric()
import os
import glob
import cv2
import random
import albumentations as A

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
DATASET_DIR = os.path.abspath('./data/YOLO/train')
IMG_DIR = os.path.join(DATASET_DIR, 'images')
LBL_DIR = os.path.join(DATASET_DIR, 'labels')

# Objetivo de balanceo por clase
TARGET_BOXES = 18100

# ==============================================================================
# 🎨 PIPELINES DE TRANSFORMACIÓN (ALBUMENTATIONS)
# ==============================================================================
geo_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, 
                       border_mode=cv2.BORDER_CONSTANT, p=1.0),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

color_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.8),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.5),
    A.Blur(blur_limit=3, p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==============================================================================
# 📊 FUNCIONES AUXILIARES
# ==============================================================================
def count_current_labels():
    """Lee los txt actuales y cuenta las instancias de cada clase."""
    counts = {0: 0, 1: 0, 2: 0}
    for file in glob.glob(os.path.join(LBL_DIR, '*.txt')):
        if file.endswith('classes.txt'): continue
        with open(file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if parts:
                    cls = int(float(parts[0]))
                    if cls in counts:
                        counts[cls] += 1
    return counts

def categorize_images():
    """Separa las rutas de las imágenes según las clases minoritarias que contengan."""
    heli_imgs, civil_imgs = [], []
    for label_path in glob.glob(os.path.join(LBL_DIR, '*.txt')):
        if label_path.endswith('classes.txt'): continue
        
        img_path = os.path.join(IMG_DIR, os.path.basename(label_path).replace('.txt', '.jpg'))
        if not os.path.exists(img_path):
            img_path = img_path.replace('.jpg', '.png')

        with open(label_path, 'r') as f:
            content = f.read()
            if '\n1 ' in content or content.startswith('1 '):
                heli_imgs.append(img_path)
            if '\n0 ' in content or content.startswith('0 '):
                civil_imgs.append(img_path)
                
    return heli_imgs, civil_imgs

# ==============================================================================
# 🚀 EJECUCIÓN DEL AUMENTO MIXTO
# ==============================================================================
def augment_mixed():
    print("🔍 Escaneando directorio y categorizando imágenes base...")
    heli_imgs, civil_imgs = categorize_images()
    print(f" -> Encontradas {len(heli_imgs)} fotos con Helicópteros y {len(civil_imgs)} con Civiles.")

    current_counts = count_current_labels()
    images_generated = 0
    telemetry = {'solo_geo': 0, 'solo_color': 0, 'combinado': 0}

    print("\n🚀 Iniciando Data Augmentation Mixto...")
    while current_counts[0] < TARGET_BOXES or current_counts[1] < TARGET_BOXES:
        
        # Priorizar la clase más deficitaria
        if current_counts[1] < TARGET_BOXES and heli_imgs:
            img_path = random.choice(heli_imgs)
        elif civil_imgs:
            img_path = random.choice(civil_imgs)
        else:
            break

        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        label_path = os.path.join(LBL_DIR, f"{name_no_ext}.txt")

        bboxes, class_labels = [], []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    class_labels.append(int(float(parts[0])))

        if not bboxes: continue

        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Selección estocástica de la ruta de aumento
            route = random.choice(['solo_geo', 'solo_color', 'combinado'])

            if route == 'solo_geo':
                transformed = geo_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            elif route == 'solo_color':
                transformed = color_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            else:
                t_inter = geo_transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed = color_transform(image=t_inter['image'], bboxes=t_inter['bboxes'], class_labels=t_inter['class_labels'])

            t_image = transformed['image']
            t_bboxes = transformed['bboxes']
            t_labels = transformed['class_labels']
            
        except Exception:
            continue

        if not t_bboxes: continue

        # Actualizar telemetría y contadores
        telemetry[route] += 1
        for c in t_labels:
            if c in current_counts:
                current_counts[c] += 1
                
        images_generated += 1

        # Guardado a disco
        new_name = f"{name_no_ext}_aug_mixto_{images_generated}"
        cv2.imwrite(os.path.join(IMG_DIR, f"{new_name}.jpg"), cv2.cvtColor(t_image, cv2.COLOR_RGB2BGR))

        with open(os.path.join(LBL_DIR, f"{new_name}.txt"), 'w') as f:
            for bbox, label in zip(t_bboxes, t_labels):
                f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        if images_generated % 500 == 0:
            print(f" -> Generadas {images_generated} imágenes... (Civil: {current_counts[0]}, Heli: {current_counts[1]})")

    # --- 📊 INFORME FINAL ---
    print("\n" + "="*50)
    print("📈 RESUMEN DE TELEMETRÍA DEL AUMENTO MIXTO")
    print("="*50)
    print(f"Total de imágenes sintéticas creadas: {images_generated}")
    if images_generated > 0:
        print(f" - Aumento puramente Geométrico:  {telemetry['solo_geo']} ({(telemetry['solo_geo']/images_generated)*100:.1f}%)")
        print(f" - Aumento puramente Fotométrico: {telemetry['solo_color']} ({(telemetry['solo_color']/images_generated)*100:.1f}%)")
        print(f" - Aumento Combinado (Geo+Color): {telemetry['combinado']} ({(telemetry['combinado']/images_generated)*100:.1f}%)")

if __name__ == "__main__":
    augment_mixed()
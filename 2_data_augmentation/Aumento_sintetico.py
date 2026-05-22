import os
import glob
import cv2
import random
import shutil
import albumentations as A

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
# Ruta del dataset principal de YOLO (donde están train, val, test)
DATASET_DIR = os.path.abspath('./data/YOLO')
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, 'train', 'images')
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, 'train', 'labels')

# Ruta de los datos sintéticos (ya filtrados de alucinaciones)
SYNTHETIC_DIR = os.path.abspath('./data/synthetic_flux')
SYNTH_IMG_DIR = os.path.join(SYNTHETIC_DIR, 'images')
SYNTH_LBL_DIR = os.path.join(SYNTHETIC_DIR, 'labels')

# Objetivos de balanceo
TARGET_CIVIL_BOXES = 17814  # Igualar aviones civiles a los militares
SPLIT_RATIO = (0.80, 0.12, 0.08)  # Train / Val / Test para los helicópteros sintéticos

# ==============================================================================
# 🎨 PIPELINES DE ALBUMENTATIONS (Para Fase 1)
# ==============================================================================
geo_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),  # Excelente para perspectiva cenital (satélite)
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=45, 
                       border_mode=cv2.BORDER_CONSTANT, p=1.0),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

color_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.8),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.5),
    A.Blur(blur_limit=3, p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==============================================================================
# 🚀 FASE 1: AUMENTO CLÁSICO (SOLO AVIONES CIVILES)
# ==============================================================================
def phase1_augment_civil_aircraft():
    print("\n" + "="*50)
    print("✈️ FASE 1: BALANCEO DE AVIONES CIVILES (Aumento Clásico)")
    print("="*50)

    civil_imgs = []
    current_civil_count = 0

    # Escanear imágenes que contienen clase 0 (Civil)
    for label_path in glob.glob(os.path.join(TRAIN_LBL_DIR, '*.txt')):
        if label_path.endswith('classes.txt'): continue
        with open(label_path, 'r') as f:
            content = f.read()
            if '\n0 ' in content or content.startswith('0 '):
                img_path = os.path.join(TRAIN_IMG_DIR, os.path.basename(label_path).replace('.txt', '.jpg'))
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.jpg', '.png')
                if os.path.exists(img_path):
                    civil_imgs.append(img_path)
                
                # Contar cajas
                current_civil_count += content.split('\n0 ').__len__() - 1
                if content.startswith('0 '): current_civil_count += 1

    print(f" -> Imágenes base con aviones civiles: {len(civil_imgs)}")
    print(f" -> Instancias actuales: {current_civil_count} / Objetivo: {TARGET_CIVIL_BOXES}")

    images_generated = 0
    while current_civil_count < TARGET_CIVIL_BOXES:
        img_path = random.choice(civil_imgs)
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        label_path = os.path.join(TRAIN_LBL_DIR, f"{name_no_ext}.txt")

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
            route = random.choice(['solo_geo', 'solo_color', 'combinado'])
            if route == 'solo_geo':
                transformed = geo_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            elif route == 'solo_color':
                transformed = color_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            else:
                t_inter = geo_transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed = color_transform(image=t_inter['image'], bboxes=t_inter['bboxes'], class_labels=t_inter['class_labels'])
            
            t_image, t_bboxes, t_labels = transformed['image'], transformed['bboxes'], transformed['class_labels']
        except Exception:
            continue

        if not t_bboxes: continue

        # Actualizar contadores
        current_civil_count += t_labels.count(0)
        images_generated += 1

        # Guardar nueva imagen
        new_name = f"{name_no_ext}_aug_civil_{images_generated}"
        cv2.imwrite(os.path.join(TRAIN_IMG_DIR, f"{new_name}.jpg"), cv2.cvtColor(t_image, cv2.COLOR_RGB2BGR))

        with open(os.path.join(TRAIN_LBL_DIR, f"{new_name}.txt"), 'w') as f:
            for bbox, label in zip(t_bboxes, t_labels):
                f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    print(f"✅ Balanceo civil completado. Se generaron {images_generated} imágenes sintéticas.")

# ==============================================================================
# 🚁 FASE 2: INYECCIÓN DE DATOS SINTÉTICOS (HELICÓPTEROS)
# ==============================================================================
def phase2_inject_synthetic_helicopters():
    print("\n" + "="*50)
    print("🚁 FASE 2: INYECCIÓN DE HELICÓPTEROS SINTÉTICOS (FLUX)")
    print("="*50)

    if not os.path.exists(SYNTH_LBL_DIR):
        print(f"⚠️ No se encontró la carpeta de etiquetas sintéticas: {SYNTH_LBL_DIR}")
        return

    valid_labels = [f for f in os.listdir(SYNTH_LBL_DIR) if f.endswith('.txt')]
    base_names = [f.replace('.txt', '') for f in valid_labels]
    
    # Fijar semilla para reproducibilidad (Mismo reparto siempre)
    random.seed(93)
    random.shuffle(base_names)

    total = len(base_names)
    idx_train = int(total * SPLIT_RATIO[0])
    idx_val = int(total * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))

    splits = {
        'train': base_names[:idx_train],
        'val': base_names[idx_train:idx_val],
        'test': base_names[idx_val:]
    }

    print(f" -> Total de imágenes sintéticas a inyectar: {total}")
    print(f" -> Reparto: Train ({len(splits['train'])}), Val ({len(splits['val'])}), Test ({len(splits['test'])})")

    for split, files in splits.items():
        dest_img_dir = os.path.join(DATASET_DIR, split, 'images')
        dest_lbl_dir = os.path.join(DATASET_DIR, split, 'labels')

        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)

        for name in files:
            # Soportar PNG o JPG de FLUX
            src_img_png = os.path.join(SYNTH_IMG_DIR, f"{name}.png")
            src_img_jpg = os.path.join(SYNTH_IMG_DIR, f"{name}.jpg")
            
            if os.path.exists(src_img_png):
                shutil.copy(src_img_png, os.path.join(dest_img_dir, f"{name}.png"))
            elif os.path.exists(src_img_jpg):
                shutil.copy(src_img_jpg, os.path.join(dest_img_dir, f"{name}.jpg"))

            shutil.copy(os.path.join(SYNTH_LBL_DIR, f"{name}.txt"), 
                        os.path.join(dest_lbl_dir, f"{name}.txt"))

    print("✅ Inyección de helicópteros sintéticos completada con éxito.")

# ==============================================================================
# 🎯 EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # 1. Igualar Aviones Civiles con Militares
    phase1_augment_civil_aircraft()
    
    # 2. Inyectar Helicópteros Sintéticos
    phase2_inject_synthetic_helicopters()
    
    print("\n🎉 ¡DATASET DEFINITIVO LISTO PARA ENTRENAMIENTO FINAL!")
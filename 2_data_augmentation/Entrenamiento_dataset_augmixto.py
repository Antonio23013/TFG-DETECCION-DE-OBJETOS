import os
import yaml
import glob
from collections import Counter
from ultralytics import YOLO

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
# Apunta al dataset que ya ha pasado por el Data Augmentation Mixto
DATASET_DIR = os.path.abspath('./data/dataset_aug_mixto_clasico')
OUTPUT_DIR = os.path.abspath('./runs/yolo_aug_clasico')
YAML_PATH = os.path.join(DATASET_DIR, 'dataset_config.yaml')

EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 🔍 FASE 1: AUDITORÍA DE BALANCEO DEL DATASET
# ==============================================================================
def verify_dataset_balance():
    """Escanea las etiquetas de entrenamiento para confirmar el balanceo de clases."""
    print("🔍 Auditando el balanceo del Dataset Aumentado...")
    
    labels_dir = os.path.join(DATASET_DIR, 'train', 'labels')
    if not os.path.exists(labels_dir):
        print(f"⚠️ No se encontró la ruta de etiquetas: {labels_dir}")
        return False

    class_counts = Counter()
    for txt_file in glob.glob(os.path.join(labels_dir, '*.txt')):
        if txt_file.endswith('classes.txt'): continue
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    cls_id = int(float(line.split()[0]))
                    class_counts[cls_id] += 1

    print("\n" + "="*50)
    print("📊 RECUENTO DE BOUNDING BOXES (DATASET MIXTO)")
    print("="*50)
    class_names = {0: 'Avión Civil', 1: 'Helicóptero', 2: 'Avión Militar'}
    
    for cls_id, name in class_names.items():
        count = class_counts.get(cls_id, 0)
        print(f" -> [{cls_id}] {name}: {count} instancias")
    print("="*50 + "\n")
    return True

# ==============================================================================
# 📝 FASE 2: GENERACIÓN DINÁMICA DEL YAML
# ==============================================================================
def create_yaml_config():
    """Genera el archivo YAML necesario para Ultralytics."""
    data_config = {
        'path': DATASET_DIR,
        'train': 'train/images',
        'val': 'val/images',  # Asegúrate de que tu partición valid se llame 'val'
        'test': 'test/images',
        'nc': 3,
        'names': {0: 'A.Civil', 1: 'Helicoptero', 2: 'A.Militar'}
    }
    with open(YAML_PATH, 'w') as f:
        yaml.dump(data_config, f)

# ==============================================================================
# 🚀 FASE 3: ENTRENAMIENTO DEL MODELO
# ==============================================================================
def train_augmented_model():
    """Ejecuta el entrenamiento de YOLO11 sobre el dataset balanceado."""
    print("🚀 INICIANDO ENTRENAMIENTO (YOLO11 Nano - Augmentation Clásico)...")
    
    model = YOLO('yolo11n.pt')
    
    model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        patience=15,  # Early stopping si no mejora en 15 épocas
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=OUTPUT_DIR,
        name=f'YOLO11n_Aug_Clasico_{IMG_SIZE}px',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        plots=True    # Genera matrices de confusión y curvas de loss automáticamente
    )
    
    print("\n🏆 Entrenamiento completado con éxito. Resultados guardados en:")
    print(os.path.join(OUTPUT_DIR, f'YOLO11n_Aug_Clasico_{IMG_SIZE}px'))

# ==============================================================================
# 🎯 EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    if verify_dataset_balance():
        create_yaml_config()
        train_augmented_model()
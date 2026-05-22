import os
import time
import yaml
import torch
import glob
from ultralytics import RTDETR

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS GENÉRICAS Y PARÁMETROS
# ==============================================================================
# Cambia DATASET_DIR al directorio local donde tengas tus datos YOLO
DATASET_DIR = os.path.abspath('./data/YOLO')
OUTPUT_DIR = os.path.abspath('./runs/rt_detr')
YAML_PATH = os.path.join(DATASET_DIR, 'dataset_config.yaml')

EPOCHS = 20
BATCH_SIZE = 8
IMG_SIZE = 640
CLOSE_MOSAIC = 6  # Épocas finales sin mosaico para refinar la precisión

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 📝 GENERACIÓN DINÁMICA DEL ARCHIVO YAML
# ==============================================================================
def create_yaml_config():
    """Genera el archivo de configuración YAML requerido por Ultralytics."""
    data_config = {
        'path': DATASET_DIR,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'A.Civil',
            1: 'Helicoptero',
            2: 'A.Militar'
        }
    }
    with open(YAML_PATH, 'w') as f:
        yaml.dump(data_config, f)
    print(f"📄 Archivo YAML de configuración generado en: {YAML_PATH}")

# ==============================================================================
# 🚀 FASE 1: ENTRENAMIENTO DEL MODELO
# ==============================================================================
def train_model():
    """Ejecuta el entrenamiento continuo de RT-DETR."""
    print(f"\n🧠 Cargando arquitectura base (RT-DETR Large)...")
    model = RTDETR('rtdetr-l.pt')

    print("\n🚀 Iniciando Entrenamiento...")
    model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        close_mosaic=CLOSE_MOSAIC,
        amp=True,  # Automatic Mixed Precision para optimizar VRAM
        project=OUTPUT_DIR,
        name='Entrenamiento_Final',
        exist_ok=True
    )
    return os.path.join(OUTPUT_DIR, 'Entrenamiento_Final', 'weights', 'best.pt')

# ==============================================================================
# 📊 FASE 2: EVALUACIÓN Y BENCHMARKING (Viabilidad On-Board)
# ==============================================================================
def evaluate_model(weights_path):
    """Analiza métricas de precisión (mAP), tamaño y velocidad (FPS)."""
    print(f"\n🔄 Cargando pesos definitivos desde: {weights_path}")
    model = RTDETR(weights_path)

    print("\n📏 ANALIZANDO ARQUITECTURA DEL MODELO")
    print("-" * 40)
    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"📦 Tamaño en Disco:   {size_mb:.2f} MB")
    model.info(detailed=False, verbose=True)  # Imprime Params y GFLOPs

    print("\n📊 EVALUANDO PRECISIÓN EN TEST")
    print("-" * 40)
    metrics = model.val(data=YAML_PATH, split='test', conf=0.25, iou=0.6, verbose=False)
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\n🏎️ MIDIENDO VELOCIDAD DE INFERENCIA (FPS)")
    print("-" * 40)
    if torch.cuda.is_available():
        test_images = glob.glob(os.path.join(DATASET_DIR, 'test', 'images', '*.jpg'))[:100]
        if not test_images:
            print("⚠️ No se encontraron imágenes en la carpeta test para el benchmark.")
            return

        print("🔥 Calentando GPU...")
        for _ in range(10):
            model.predict(test_images[0], verbose=False, save=False)

        print("⏱️ Midiendo...")
        start_time = time.time()
        for img in test_images:
            model.predict(img, verbose=False, save=False, conf=0.25)
        total_time = time.time() - start_time
        
        avg_time_per_img = total_time / len(test_images)
        fps = 1 / avg_time_per_img

        print(f"Imágenes procesadas: {len(test_images)}")
        print(f"Latencia media:      {avg_time_per_img * 1000:.2f} ms/img")
        print(f"FPS Estimados:       {fps:.2f} FPS")
    else:
        print("⚠️ No se detecta GPU. Benchmark de FPS cancelado para evitar falsos negativos en CPU.")

# ==============================================================================
# 🎯 EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🖥️ Entorno de ejecución: {device}")
    
    create_yaml_config()
    
    # 1. Entrenar
    best_weights = train_model()
    
    # 2. Evaluar
    evaluate_model(best_weights)
    
    print("\n✅ Proceso de RT-DETR completado.")
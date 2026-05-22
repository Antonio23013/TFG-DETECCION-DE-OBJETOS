import os
import time
import yaml
import torch
import glob
from ultralytics import YOLO

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS GENÉRICAS Y PARÁMETROS
# ==============================================================================
# Ajusta estas rutas a tu entorno local
DATASET_DIR = os.path.abspath('./data/YOLO')
OUTPUT_DIR = os.path.abspath('./runs/yolo')
YAML_PATH = os.path.join(DATASET_DIR, 'dataset_config.yaml')

# Parámetros del modelo (YOLO11 Nano)
MODEL_NAME = 'yolo11n.pt'
EPOCHS = 20
BATCH_SIZE = 16       # Ajustable según la VRAM disponible
IMG_SIZE = 1024       # Resolución de entrenamiento e inferencia

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
        'nc': 3,
        'names': {
            0: 'A.Civil',
            1: 'Helicoptero',
            2: 'A.Militar'
        }
    }
    with open(YAML_PATH, 'w') as f:
        yaml.dump(data_config, f)
    print(f"📄 Archivo YAML generado en: {YAML_PATH}")

# ==============================================================================
# 🚀 FASE 1: ENTRENAMIENTO DEL MODELO
# ==============================================================================
def train_model():
    """Ejecuta el entrenamiento del modelo YOLO."""
    print(f"\n🧠 Cargando arquitectura base ({MODEL_NAME})...")
    model = YOLO(MODEL_NAME)

    print("\n🚀 Iniciando Entrenamiento...")
    model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=OUTPUT_DIR,
        name=f'YOLO11n_{EPOCHS}E_{IMG_SIZE}px',
        pretrained=True,
        optimizer='auto',
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    weights_path = os.path.join(OUTPUT_DIR, f'YOLO11n_{EPOCHS}E_{IMG_SIZE}px', 'weights', 'best.pt')
    return weights_path

# ==============================================================================
# 📊 FASE 2: EVALUACIÓN Y BENCHMARKING (Viabilidad On-Board)
# ==============================================================================
def evaluate_model(weights_path):
    """Analiza métricas de precisión (mAP), carga estructural y velocidad (FPS)."""
    print(f"\n🔄 Cargando pesos definitivos desde: {weights_path}")
    model = YOLO(weights_path)

    print("\n📏 ANALIZANDO ARQUITECTURA DEL MODELO (SWaP)")
    print("-" * 40)
    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"📦 Tamaño en Disco:   {size_mb:.2f} MB")
    
    # Cálculo geométrico de GFLOPs según resolución (YOLO11n base 640px = 6.4 GFLOPs)
    gflops_base_640 = 6.4
    factor_area = (IMG_SIZE * IMG_SIZE) / (640 * 640)
    gflops_reales = gflops_base_640 * factor_area
    print(f"🧠 Complejidad:       ~2.59 Millones Params | {gflops_reales:.1f} GFLOPs (@{IMG_SIZE}px)")

    print("\n📊 EVALUANDO PRECISIÓN EN TEST")
    print("-" * 40)
    metrics = model.val(
        data=YAML_PATH, 
        split='test', 
        imgsz=IMG_SIZE, 
        batch=1, 
        conf=0.25, 
        iou=0.6, 
        plots=False, 
        verbose=False
    )
    
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
            model.predict(test_images[0], verbose=False, imgsz=IMG_SIZE, save=False)

        print("⏱️ Midiendo...")
        start_time = time.time()
        for img in test_images:
            model.predict(img, verbose=False, save=False, imgsz=IMG_SIZE, conf=0.25)
        total_time = time.time() - start_time
        
        avg_time_per_img = total_time / len(test_images)
        fps = 1 / avg_time_per_img

        print(f"Imágenes procesadas: {len(test_images)}")
        print(f"Latencia media:      {avg_time_per_img * 1000:.2f} ms/img")
        print(f"FPS Estimados:       {fps:.2f} FPS")
    else:
        print("⚠️ No se detecta GPU. Benchmark de FPS cancelado para evitar datos irreales en CPU.")

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
    
    print("\n✅ Proceso de YOLO completado.")
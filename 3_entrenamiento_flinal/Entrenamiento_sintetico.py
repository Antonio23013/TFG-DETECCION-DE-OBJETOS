import os
from ultralytics import YOLO

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
# Apuntamos a la carpeta donde el script anterior dejó los datos ya mezclados
DATASET_DIR = os.path.abspath('./data/YOLO')
YAML_PATH = os.path.join(DATASET_DIR, 'dataset_config.yaml')
OUTPUT_DIR = os.path.abspath('./runs/final_training')

MODEL_NAME = 'yolo11n.pt'
EPOCHS = 20
IMG_SIZE = 640
BATCH_SIZE = 16   # Ajustable a 8 si hay limitaciones de VRAM en la GPU
SEED = 93         # Semilla fijada para garantizar reproducibilidad exacta

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 🚀 ENTRENAMIENTO DEFINITIVO (DATOS REALES + SINTÉTICOS)
# ==============================================================================
def train_final_model():
    print("\n" + "="*55)
    print("🏆 INICIANDO ENTRENAMIENTO DEFINITIVO (YOLO11 Nano)")
    print("="*55)
    
    if not os.path.exists(YAML_PATH):
        print(f"❌ ERROR: No se encuentra el archivo de configuración {YAML_PATH}")
        print("Asegúrate de haber ejecutado los scripts de la Fase 1 y 2 primero.")
        return
        
    print(f"🧠 Cargando arquitectura base: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    print("🚀 Lanzando entrenamiento con el dataset 100% balanceado...")
    model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        project=OUTPUT_DIR,
        name='YOLO11n_Final_Balanced',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        plots=True   # Generará automáticamente las matrices de confusión y gráficas P-R
    )
    
    print("\n" + "="*55)
    print("✅ ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
    final_path = os.path.join(OUTPUT_DIR, 'YOLO11n_Final_Balanced')
    print(f"📁 Los pesos definitivos (best.pt) y métricas están en: \n{final_path}")
    print("="*55)

if __name__ == "__main__":
    train_final_model()
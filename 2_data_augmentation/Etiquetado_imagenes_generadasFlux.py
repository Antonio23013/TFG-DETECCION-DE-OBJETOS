import os
import shutil
import glob
from ultralytics import YOLO

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
# Ruta al mejor modelo entrenado en la Fase 2 (Aumento Clásico)
MODEL_PATH = os.path.abspath('./runs/yolo_aug_clasico/weights/best.pt')

# Directorios de los datos sintéticos generados por FLUX/LoRA
SYNTHETIC_DIR = os.path.abspath('./data/synthetic_flux')
IMG_DIR = os.path.join(SYNTHETIC_DIR, 'images')
LBL_DIR = os.path.join(SYNTHETIC_DIR, 'labels')
DISCARD_DIR = os.path.join(SYNTHETIC_DIR, 'discarded_hallucinations')

# Configuración del filtro estricto
TARGET_CLASS_ID = 1      # ID 1 = Helicóptero (Ajustar según data.yaml si varía)
CONF_THRESHOLD = 0.4     # Umbral de confianza exigido para validar la imagen

# Crear directorios de salida
os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(DISCARD_DIR, exist_ok=True)

# ==============================================================================
# 🚀 AUTO-ETIQUETADO Y CONTROL DE CALIDAD (QA)
# ==============================================================================
def auto_label_and_filter():
    """
    Utiliza el modelo baseline para etiquetar automáticamente imágenes sintéticas.
    Descarta aquellas donde el modelo de difusión ha fallado (alucinaciones).
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: No se encontró el modelo en {MODEL_PATH}")
        return

    print("🚀 Cargando modelo auditor (Baseline Fase 2)...")
    model = YOLO(MODEL_PATH)

    images = [f for f in glob.glob(os.path.join(IMG_DIR, '*.*')) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"⚠️ No se encontraron imágenes sintéticas en {IMG_DIR}")
        return

    print(f"🔍 Iniciando escaneo y filtrado QA de {len(images)} imágenes sintéticas...")
    
    success_count = 0
    discard_count = 0

    for img_path in images:
        img_name = os.path.basename(img_path)
        
        # Inferencia silenciosa
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        valid_labels = []

        # Filtrar detecciones
        for box in results.boxes:
            cls_id = int(box.cls[0])
            
            # FILTRO ESTRICTO: Solo aceptamos la clase objetivo (Helicóptero)
            if cls_id == TARGET_CLASS_ID:
                x, y, w, h = box.xywhn[0].cpu().numpy()
                valid_labels.append(f"{TARGET_CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # Gestión de archivos basada en el control de calidad
        if valid_labels:
            # ÉXITO: Se detectó correctamente, guardamos las coordenadas
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            with open(os.path.join(LBL_DIR, txt_name), 'w') as f:
                f.write("\n".join(valid_labels))
            success_count += 1
        else:
            # ANOMALÍA: FLUX alucinó o deformó la aeronave. Se descarta la imagen.
            shutil.move(img_path, os.path.join(DISCARD_DIR, img_name))
            discard_count += 1

    # ==============================================================================
    # 📊 TELEMETRÍA FINAL
    # ==============================================================================
    print("\n" + "="*50)
    print("📈 REPORTE DE CALIDAD (SYNTHETIC DATA QA)")
    print("="*50)
    print(f"✅ Imágenes válidas auto-etiquetadas: {success_count}")
    print(f"🗑️ Imágenes descartadas (Alucinaciones): {discard_count}")
    
    if len(images) > 0:
        retention_rate = (success_count / len(images)) * 100
        print(f"🎯 Tasa de retención de datos sintéticos: {retention_rate:.1f}%")

if __name__ == "__main__":
    auto_label_and_filter()
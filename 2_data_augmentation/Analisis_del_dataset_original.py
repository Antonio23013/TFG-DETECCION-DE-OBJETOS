import os
import glob
import matplotlib.pyplot as plt

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS GENÉRICAS
# ==============================================================================
# Apunta a la carpeta donde tienes el dataset extraído
DATASET_DIR = os.path.abspath('./data/YOLO')
OUTPUT_DIR = os.path.abspath('./runs/dataset_analysis')

CLASS_NAMES = {
    0: 'Avión Civil',
    1: 'Helicóptero',
    2: 'Avión Militar'
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 🔍 ANÁLISIS DEL DATASET (Conteo y Morfología)
# ==============================================================================
def analyze_dataset():
    """Analiza el desbalance de clases y el tamaño relativo de las aeronaves."""
    print(f"🔍 Analizando dataset en: {DATASET_DIR}")
    
    class_counts = {0: 0, 1: 0, 2: 0}
    class_areas = {0: [], 1: [], 2: []}

    # Buscar etiquetas exclusivamente en la partición de entrenamiento
    labels_path = os.path.join(DATASET_DIR, 'train', 'labels', '*.txt')
    txt_files = [f for f in glob.glob(labels_path) if not f.endswith('classes.txt')]
    
    if not txt_files:
        print("⚠️ No se encontraron archivos .txt en la carpeta train/labels. Verifica la ruta.")
        return

    print(f"📂 Procesando {len(txt_files)} archivos de etiquetas...")

    # Lectura unificada: Sacamos conteo y área en la misma pasada (Más eficiente)
    for file in txt_files:
        with open(file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    w, h = float(parts[3]), float(parts[4])
                    
                    area_pct = (w * h) * 100  # Área relativa en porcentaje
                    
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
                        class_areas[cls_id].append(area_pct)

    # --- 📊 RESULTADOS NUMÉRICOS EN CONSOLA ---
    total_boxes = sum(class_counts.values())
    print("\n" + "="*50)
    print("📈 DIAGNÓSTICO DEL DATASET ORIGINAL (TRAIN)")
    print("="*50)
    print(f"Total de instancias (Bounding Boxes): {total_boxes}\n")

    for cls_id, count in class_counts.items():
        pct = (count / total_boxes) * 100 if total_boxes > 0 else 0
        areas = class_areas[cls_id]
        
        print(f"✈️  {CLASS_NAMES[cls_id]}:")
        print(f"   - Cantidad:   {count} instancias ({pct:.2f}% del total)")
        
        if areas:
            avg_area = sum(areas) / len(areas)
            small_objs = sum(1 for a in areas if a < 1.0)
            pct_small = (small_objs / len(areas)) * 100
            print(f"   - Área media: {avg_area:.3f}% de la imagen")
            print(f"   - Micro-objs: {pct_small:.1f}% miden menos del 1% del encuadre\n")

    # Generar visualizaciones
    generate_plots(class_counts, class_areas)

# ==============================================================================
# 📉 GENERACIÓN DE GRÁFICAS DE ALTA DEFINICIÓN
# ==============================================================================
def generate_plots(counts, areas):
    """Genera y guarda las gráficas de barras y boxplots en disco."""
    colors = ['#3498db', '#e67e22', '#2ecc71'] # Azul, Naranja, Verde
    labels = [CLASS_NAMES[i] for i in range(3)]
    
    # 1. Gráfica de Barras (Desbalance de Clases)
    plt.figure(figsize=(8, 6))
    values = [counts[i] for i in range(3)]
    bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01),
                 int(yval), ha='center', va='bottom', fontweight='bold')

    plt.title('Distribución de Clases (Desbalance Original)', fontsize=14, fontweight='bold')
    plt.ylabel('Número de Instancias', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_distribucion_clases.png'), dpi=300)
    plt.close()

    # 2. Boxplot (Morfología y Tamaño)
    plt.figure(figsize=(9, 6))
    plot_data = [areas[i] for i in range(3)]
    
    box = plt.boxplot(plot_data, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

    plt.title('Morfología: Área Relativa de las Aeronaves', fontsize=14, fontweight='bold')
    plt.ylabel('Área de la Bounding Box (% de la imagen)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_distribucion_areas.png'), dpi=300)
    plt.close()

    print(f"✅ Gráficas de diagnóstico guardadas exitosamente en: {OUTPUT_DIR}")

# ==============================================================================
# 🎯 EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    analyze_dataset()
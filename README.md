<div align="center">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Satellite.png" width="120" alt="Satellite"/>

# Detección de Aeronaves en Imágenes Satelitales para CubeSats

**Trabajo de Fin de Grado · Ingeniería Aeroespacial**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

*Sistema de Visión Artificial optimizado para teledetección orbital. Mitigación de desbalance de clases mediante técnicas clásicas y generación de datos sintéticos (FLUX/LoRA).*

---

</div>

## 🌌 Resumen del Proyecto

Este repositorio aloja el código fuente y la metodología analítica de un sistema de detección de aeronaves diseñado específicamente para su despliegue en plataformas espaciales de baja órbita (CubeSats). 

En el ámbito de la Inteligencia Geoespacial (GEOINT) y las operaciones ISR (*Intelligence, Surveillance, and Reconnaissance*), el procesamiento de imágenes satelitales se enfrenta a dos cuellos de botella tecnológicos que este proyecto resuelve:

1. **Escasez Crítica de Datos:** Las clases minoritarias, como helicópteros o aeronaves de combate, presentan un desbalance extremo frente a la aviación comercial.
2. **Restricciones SWaP (*Size, Weight, and Power*):** El hardware embarcado en un nanosatélite exige modelos de inferencia ultraligeros, con latencias mínimas y bajo consumo energético.

---

## 🔬 Arquitecturas de Inferencia Evaluadas

Para determinar la viabilidad tecnológica *on-board*, se ha diseñado un banco de pruebas comparativo entre tres enfoques líderes en visión artificial:

| Familia de Modelo | Arquitectura Base | Propósito en el Estudio |
| :--- | :--- | :--- |
| **Ultralytics YOLO** | *One-stage anchor-free* | Evaluar el mejor equilibrio histórico entre precisión (mAP) y tiempos de latencia. |
| **RT-DETR** | *Vision Transformer* (ViT) | Analizar la viabilidad de mecanismos de atención espacial eliminando la dependencia del *Non-Maximum Suppression* (NMS). |
| **MobileNet V3** | *Lightweight CNN* | Establecer la línea base de mínimo consumo computacional para aceleradores de IA en el *Edge*. |

---

## 🗂️ Estructura Detallada del Repositorio

El código está segmentado de forma modular, reflejando el *pipeline* completo de ingeniería de datos y entrenamiento:

### 📁 `1_modelo_evaluacion/` (Fase Base)
*Scripts* utilizados para el entrenamiento inicial sobre el conjunto de datos crudo y desbalanceado. Establecen las métricas de referencia.
* `YOLO.py`: *Pipeline* de entrenamiento y benchmarking para la arquitectura YOLO.
* `RT-DETR.py`: Adaptación y entrenamiento del modelo basado en *Transformers*.
* `SSD_MobileNet.py`: Configuración de la red convolucional ligera y ajuste de normalización.

### 📁 `2_data_augmentation/` (Ingeniería de Datos y QA)
El núcleo de la solución al desbalance. Incluye tanto transformaciones matemáticas tradicionales como el filtrado de imágenes generadas por Inteligencia Artificial.
* `Analisis_del_dataset_original.py`: Script de Análisis Exploratorio (EDA) para diagnosticar morfológicamente el desbalance inicial.
* `Aumento_geometrico.py` / `Aumento_fotometrico.py`: Algoritmos aislados de alteración espacial y de color.
* `Aumento_mixto_tecnicas_clasicas.py`: *Pipeline* principal que automatiza el balanceo de aviones civiles generando un subconjunto equilibrado de **2832 imágenes**.
* `Etiquetado_imagenes_generadasFlux.py`: Sistema de Control de Calidad (QA) y auto-etiquetado. Usa un modelo auditor para descartar alucinaciones visuales del modelo de difusión FLUX.
* `Aumento_sintetico.py`: Lógica de inyección y mezcla que reparte los helicópteros sintéticos en las particiones de entrenamiento y validación.
* `Entrenamiento_dataset_augmixto.py`: Validación empírica para confirmar la mejora de precisión tras el aumento clásico.

### 📁 `3_entrenamiento_flinal/` (Despliegue Definitivo)
* `Entrenamiento_sintetico.py`: *Script* maestro de entrenamiento. Ingiere el *dataset* 100% balanceado (Reales + Clásicos + Sintéticos) para generar los pesos definitivos y las métricas finales del TFG.

---

## 🚀 Instalación y Reproducibilidad

El entorno de trabajo está diseñado para ser portable, compatible con ejecución en CPU/GPU local o instancias *cloud* como Google Colab.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/Antonio23013/TFG-DETECCION-DE-OBJETOS.git](https://github.com/Antonio23013/TFG-DETECCION-DE-OBJETOS.git)
cd TFG-DETECCION-DE-OBJETOS
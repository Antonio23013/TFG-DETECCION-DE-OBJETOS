<div align="center">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Satellite.png" width="120" alt="Satellite"/>

# Detección de Aeronaves en Imágenes Satelitales para CubeSats

**Trabajo de Fin de Grado · Ingeniería Aeroespacial**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

*Sistema de Visión Artificial optimizado para teledetección orbital. Mitigación de desbalance de clases mediante generación de datos sintéticos (LoRA).*

---

</div>

## 🌌 Resumen del Proyecto

Este repositorio aloja el código fuente y la metodología analítica de un sistema de detección de aeronaves diseñado específicamente para su despliegue en plataformas espaciales de baja órbita (CubeSats). 

En el ámbito de la Inteligencia Geoespacial (GEOINT) y las operaciones ISR (Intelligence, Surveillance, and Reconnaissance), el procesamiento de imágenes satelitales se enfrenta a dos cuellos de botella tecnológicos que este proyecto resuelve:

1. **Escasez Crítica de Datos:** Las clases minoritarias, como helicópteros o aeronaves de combate, presentan un desbalance extremo frente a la aviación comercial.
2. **Restricciones SWaP (Size, Weight, and Power):** El hardware embarcado en un nanosatélite exige modelos de inferencia ultraligeros, con latencias mínimas y bajo consumo energético.

---

## 🔬 Arquitecturas de Inferencia Evaluadas

Para determinar la viabilidad tecnológica *on-board*, se ha diseñado un banco de pruebas comparativo entre tres enfoques líderes en visión artificial:

| Familia de Modelo | Arquitectura Base | Propósito en el Estudio |
| :--- | :--- | :--- |
| **Ultralytics YOLO** | One-stage anchor-free | Evaluar el mejor equilibrio histórico entre *Mean Average Precision* (mAP) y tiempos de latencia. |
| **RT-DETR** | Vision Transformer (ViT) | Analizar la viabilidad de mecanismos de atención espacial eliminando la dependencia del *Non-Maximum Suppression* (NMS). |
| **MobileNet V3** | Lightweight CNN | Establecer la línea base de mínimo consumo computacional para aceleradores de IA en el *Edge*. |

---

## 🗂️ Estructura Detallada del Repositorio

El código está segmentado de forma modular, reflejando las tres grandes fases de la investigación:

### 📁 `1_model_evaluation/` (Fase Base)
Contiene los *scripts* utilizados para el entrenamiento inicial sobre el conjunto de datos crudo (desbalanceado). Sirve para extraer las métricas de referencia.
* `train_mobilenet_v3.py`: Configuración y entrenamiento de la red convolucional ligera.
* `train_yolo.py`: *Pipeline* de entrenamiento para YOLO, incluyendo configuración de hiperparámetros.
* `train_rtdetr.py`: Adaptación y entrenamiento del modelo basado en *Transformers*.

### 📁 `2_data_augmentation/` (Fase de Procesamiento Clásico)
Implementación de la ingeniería de datos tradicional para robustecer el modelo frente a variaciones atmosféricas y de perspectiva orbital.
* `augmentation_pipeline.py`: Algoritmo principal que automatiza el aumento de datos. Genera un *dataset* equilibrado de **2832 imágenes** aplicando una estrategia tripartita:
  * **Aumento Geométrico:** Rotaciones, recortes y perspectiva (32.8% - 928 imágenes).
  * **Aumento Fotométrico:** Corrección gamma, ruido gaussiano y variaciones de exposición (33.2% - 939 imágenes).
  * **Aumento Combinado:** Fusión de técnicas geométricas y de color (34.1% - 965 imágenes).

### 📁 `3_final_training_synthetic/` (Fase de IA Generativa)
El núcleo de la solución al desbalance. Integración de datos sintéticos generados mediante modelos de difusión.
* `train_final_model.py`: *Script* de entrenamiento definitivo. Ingiere el conjunto de datos enriquecido con imágenes sintéticas de helicópteros y cazas, generadas previamente mediante la técnica matemática **LoRA (Low-Rank Adaptation)**. Al inyectar matrices de bajo rango, la red aprende morfologías complejas sin el coste de un reentrenamiento masivo.
* `val_final_model.py`: Herramienta de validación para extraer curvas de precisión, *Recall* y matrices de confusión del modelo propuesto.

---

## 🚀 Instalación y Despliegue

El entorno de trabajo está diseñado para ser portable y reproducible, compatible con ejecución en CPU/GPU local o instancias *cloud* como Google Colab.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TuUsuario/NombreDelRepositorio.git](https://github.com/TuUsuario/NombreDelRepositorio.git)
cd NombreDelRepositorio
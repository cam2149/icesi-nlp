# NLP Basics Assessment: Extracción de Sentimiento de Tweets

## Descripción General
Este notebook Jupyter realiza un análisis básico de procesamiento de lenguaje natural (NLP) enfocado en la extracción de sentimiento de tweets. El objetivo es identificar el sentimiento (positivo, negativo o neutral) en tweets y extraer las palabras o frases que lo respaldan. Se basa en un conjunto de datos de Kaggle y aplica técnicas desde reglas simples hasta modelos de machine learning y transformers.

El notebook es parte de la Maestría en Inteligencia Artificial Aplicada de ICESI, desarrollado por:
- Angelica Maria Mayor
- Freddy Mauricio Gutierrez
- Wilman Quiñonez
- Carlos Alberto Martinez Ramirez
- Diego Agudelo

**Referencias clave:**
- Competencia Kaggle: [Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)
- Licencia: Creative Commons Attribution 4.0 International
- Nota: El dataset contiene texto que puede ser profano o ofensivo.

## Requisitos y Dependencias
- **Entorno:** Python 3.11+ (probado en Google Colab).
- **Instalaciones requeridas:**
  - `!pip install kaggle vaderSentiment tqdm nltk spacy GingerIt numpy==1.24.4 scikit-learn==1.2.2`
  - Modelos SpaCy: `!python -m spacy download en_core_web_trf`
  - NLTK: `nltk.download('vader_lexicon')`
  - Otras bibliotecas: pandas, matplotlib, seaborn, statsmodels, torch, lightning, tensorboard, bokeh, transformers, datasets, torchinfo, accelerate, evaluate, sentence-transformers, gradio, ollama.
- **Dataset:** Descargado de Kaggle (`tweet-sentiment-extraction.zip`). Requiere credenciales de Kaggle (archivo `kaggle.json`).

## Estructura del Notebook
El notebook se divide en secciones clave:

1. **Instalación y Configuración Inicial:**
   - Instalación de paquetes y descarga de modelos (SpaCy, NLTK).
   - Descarga condicional del dataset de Kaggle si no existe localmente.

2. **Carga y Exploración de Datos:**
   - Carga del dataset (`train.csv`) con columnas: `textID`, `text`, `selected_text`, `sentiment`.
   - Limpieza básica: Eliminación de NaN, normalización de etiquetas de sentimiento (`negative`, `neutral`, `positive`).
   - Exploración: Vista previa, conteo de oraciones con SpaCy, visualización de dependencias sintácticas usando `displacy`.

3. **Extracción de Justificación (Matcher con SpaCy):**
   - Uso de `Matcher` para identificar palabras positivas (e.g., "good", "love") y negativas (e.g., "bad", "sad", "bullying").
   - Extracción de frases clave que justifican el sentimiento.
   - Barra de progreso con `tqdm` para procesar el dataset.

4. **Preprocesamiento de Texto:**
   - Lematización y eliminación de stopwords/puntuación con SpaCy.
   - Columna generada: `processed_text`.

5. **Análisis de Sentimiento con VADER:**
   - Aplicación de `SentimentIntensityAnalyzer` para obtener scores (`neg`, `neu`, `pos`, `compound`).
   - Predicción de sentimiento basada en thresholds (`compound >= 0.05` → positive, etc.).
   - Evaluación: Accuracy (~0.6324), F1-macro, reporte de clasificación y matriz de confusión.
   - Porcentaje de coincidencia entre sentimiento real y predicho (~63.24%).

6. **Modelos Avanzados de Clasificación:**
   - **TF-IDF + LinearSVC:**
     - Vectorización con TF-IDF (ngrams=1-3, max_features=80000).
     - Entrenamiento con LinearSVC (balanced classes).
     - Resultados: Accuracy (~0.69), F1-macro (~0.69).
   - **RoBERTa Transformer (Inferencia directa):**
     - Modelo: `cardiffnlp/twitter-roberta-base-sentiment-latest`.
     - Tokenización y predicción en batches.
     - Resultados: Accuracy (~0.72), F1-macro (~0.72) – el mejor rendimiento.
   - **TF-IDF + LogisticRegression:**
     - Grid search en C=[0.5,1.0,2.0].
     - Resultados: Accuracy (~0.6941), F1-macro (~0.6954) para C=0.5.
   - **SBERT + LogisticRegression (Opcional, desactivado por defecto):**
     - Embeddings con `all-MiniLM-L6-v2`.
     - Similar a LogisticRegression, pero con vectores densos.

7. **Evaluación y Comparación:**
   - Métricas: Accuracy, F1-macro, reportes de clasificación y matrices de confusión para cada modelo.
   - Resumen comparativo en tabla (e.g., RoBERTa > TF-IDF+LogReg > VADER).
   - Predicciones almacenadas en columnas como `pred_svm`, `pred_transformer`, `pred_tfidf_logreg`.

## Resultados Principales
- **Distribución de clases:** Neutral (~10326), Positive (~7992), Negative (~7096).
- **Mejores métricas:**
  - VADER: Accuracy=0.6324, F1-macro=0.63.
  - TF-IDF + LinearSVC: Accuracy=0.69, F1-macro=0.69.
  - RoBERTa: Accuracy=0.72, F1-macro=0.72 (mejor modelo, entiende contexto mejor).
  - TF-IDF + LogReg: Accuracy=0.6941, F1-macro=0.6954.
- Observaciones: Modelos clásicos son rápidos pero limitados en contexto; transformers destacan en matices pero son más pesados.

## Cómo Ejecutar
1. Clona el repositorio o abre en Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cam2149/icesi-nlp/blob/main/Sesion1/8_practice.ipynb)
2. Sube `kaggle.json` para descargar el dataset.
3. Ejecuta las celdas secuencialmente (puede requerir restart del kernel tras instalaciones).
4. Opcional: Activa `RUN_TRANSFORMER=True` o `RUN_SBERT=True` para modelos pesados (requiere GPU/TPU).

## Limitaciones y Mejoras
- Dataset desbalanceado; considerar oversampling.
- No se entrena RoBERTa (solo inferencia); fine-tuning podría mejorar.
- Pruebas en entornos sin GPU pueden fallar en transformers.
- Posibles mejoras: Integrar más patrones en Matcher, usar ensembles de modelos.

## Licencia
Basado en dataset bajo Creative Commons Attribution 4.0. Código bajo licencia open-source implícita (ver repositorio original).

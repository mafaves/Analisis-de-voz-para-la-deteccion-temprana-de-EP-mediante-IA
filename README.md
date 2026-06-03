# Análisis de Voz para la Detección de la Enfermedad de Parkinson

Código del Trabajo de Fin de Máster — TFM_code

## Descripción

Este repositorio contiene el código para analizar grabaciones de voz y detectar la enfermedad de Parkinson (PD), así como clasificar distintos estadios. Se comparan enfoques de Machine Learning (ML) y Deep Learning (DL).

La base de datos consta de cuatro grupos de pacientes:

| Grupo | Descripción | Etiqueta |
|-------|-------------|----------|
| HC | Healthy Control (control sano) | 0 |
| NFC | Negative Family Carrier (portador familiar negativo) | 0 |
| AC | Asymptomatic Carrier (portador asintomático de la mutación G2019S **LRRK2**) | 1 |
| PD | Parkinson's Disease | 2 |

NFC se trata como HC a efectos de clasificación binaria.

## Estructura de los datos de audio

Los audios originales deben organizarse en `data/raw/` siguiendo esta estructura:

```
data/raw/
├── HC/
│   └── HUMV_HC_001/
│       └── vocal.wav
├── NFC/
│   └── HUMV_NFC_001/
│       └── vocal.wav
├── AC/
│   └── HUMV_AC_001/
│       └── vocal.wav
└── PD/
    └── HUMV_PD_001/
        └── vocal.wav
```

Cada paciente tiene su propia carpeta con uno o varios archivos de audio (p. ej., `vocal.wav`). La función `humv_loader.load_audio_data()` recorre esta estructura y devuelve un DataFrame con las columnas:

- `Patient`: ID del paciente (ej. `HUMV_PD_001`).
- `Label`: etiqueta numérica (0, 1, 2).
- `File_Path`: ruta completa al archivo de audio.
- `Audio_Name`: nombre del archivo sin extensión.

## Flujo de trabajo

### 0. Preprocesado inicial (opcional)

Estos notebooks se encuentran en `src/utilities/` y solo es necesario ejecutarlos una vez:

- **`DeepFilter_code.ipynb`** — Filtrar ruido de los audios mediante [DeepFilterNet](https://github.com/rikorose/deepfilternet).
- **`Metadata_analysis_clean.ipynb`** — Analizar variables demográficas (edad, sexo, años de educación) para validar la homogeneidad entre grupos.

### 1. Generar fragmentos de audio (chunks)

**Notebook:** `src/data/save_pre_processed_data.ipynb`

Los audios originales suelen ser largos. Para acelerar el entrenamiento, se dividen en fragmentos más cortos (5s o 10s, con o sin solapamiento de 1 s).

```python
from preprocessing import audio_processor

audio_chunks, labels, patient_ids, exercises = audio_processor.execute_preprocess_and_split(
    df, start_time=0, chunk_duration=5, max_duration=15,
    target_sr=16000, remove_silence=True, top_db=20, overlap = 0
)
```

Los resultados se guardan como archivos `.npy` en `data/processed/`:

```
data/processed/
├── 5s_with_1s_overlap_16kHz_top_db_20/
├── 5s_with_no_overlap_16kHz_top_db_20/
├── 10s_with_1s_overlap_16kHz_top_db_20/
└── 10s_with_no_overlap_16kHz_top_db_20/
```

Esto evita tener que reprocesar los audios cada vez.

### 2. Cargar datos preprocesados

**Módulo:** `src/data/load_preprocessed_data.py`

```python
from data.load_preprocessed_data import load_preprocessed_data

data_dict = load_preprocessed_data(
    processed_folder='data/processed/5s_with_1s_overlap_16kHz_top_db_20',
    pattern_type='audio_segments_5s_with_1s_overlap_*.npy'
)
# Devuelve un dict con: audio_segments, labels, patient_ids, exercises
```

### 3. Extraer características (solo para ML)

Para modelos ML se extraen features con alguno de los siguientes extractores en `src/features/`:

| Extractor | Descripción |
|-----------|-------------|
| **OpenSMILE** | ComParE 2016, eGeMAPS — características acústicas estándar |
| **Praat** | Pitch, formantes, jitter, shimmer, HNR — características fonatorias |
| **Librosa** | MFCCs, contraste espectral — características de timbre |

Para Deep Learning los chunks de audio se pasan directamente y el dataloader (`src/dataloader/audio_dataset_class.py`) genera espectrogramas Mel sobre la marcha.

### 4. Entrenar modelos

#### Machine Learning

**Notebook:** `src/run_ML_models.ipynb`

Modelos disponibles:
- **Random Forest**: conjunto de árboles de decisión
- **SVM**: máquina de vectores de soporte
- **XGBoost**: gradiente potenciado

Usa `SklearnTrainer` de `src/training/sklearn_trainer.py` que implementa validación cruzada con `StratifiedGroupKFold` para evitar fuga de datos.

#### Deep Learning

**Notebook:** `src/run_DL_models.ipynb`

Modelos disponibles en `src/models/`:
- **CNN / Bi-LSTM**: redes convolucionales y recurrentes clásicas
- **ResNet, EfficientNet**: transfer learning desde ImageNet
- **AST (Audio Spectrogram Transformer)**: transformer aplicado a espectrogramas

Arquitecturas definidas en:
- `Models.py` — CNN1D, CNN2D, Bi-LSTM
- `HigherModels.py` — ResNet, EfficientNet
- `ast_models.py` — Audio Spectrogram Transformer

### Prevención de fuga de datos (data leakage)

Todos los splits de train/test se hacen a nivel de paciente. El `StratifiedGroupKFold` de sklearn (con grupos = patient_ids) asegura que ningún paciente aparezca simultáneamente en train y test, lo cual es crítico en datos médicos con múltiples grabaciones por paciente.

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
    ...
```

## Métricas

Las predicciones a nivel de fragmento se agregan a nivel de paciente mediante `calculate_patient_wise_metrics()` en `src/utilities/stats.py`. Se calculan:

- Accuracy, precisión, recall, F1-score
- AUC-ROC
- Matriz de confusión

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura del repositorio

```
TFM_code/
├── src/
│   ├── data/              # Carga de datos
│   │   ├── humv_loader.py
│   │   └── load_preprocessed_data.py
│   ├── preprocessing/     # División en fragmentos
│   │   └── audio_processor.py
│   ├── features/          # Extracción de características
│   │   ├── opensmile.py
│   │   ├── praat.py
│   │   └── librosa_features.py
│   ├── models/            # Definición de modelos
│   │   ├── Models.py
│   │   ├── HigherModels.py
│   │   ├── ast_models.py
│   │   └── sklearn/       # Wrappers sklearn
│   ├── dataloader/        # Dataset PyTorch
│   │   └── audio_dataset_class.py
│   ├── training/          # Pipelines de entrenamiento
│   │   └── sklearn_trainer.py
│   ├── utilities/         # Utilidades (stats, notebooks auxiliares)
│   ├── analysis/          # Resumen de resultados
│   ├── run_ML_models.ipynb
│   ├── run_DL_models.ipynb
│   └── traintest_without_GRL.py
├── data/
│   ├── raw/               # Audios originales (gitignored)
│   ├── processed/         # Audios fragmentados (gitignored)
│   └── features/          # Features extraídas (gitignored)
└── outputs/
    └── experiments/       # Resultados de experimentos (gitignored)
```

## Resultados

Los resultados de cada experimento se guardan en `outputs/experiments/` con la estructura de carpetas `<comparacion>/<modelo>/`. Para consultar un resumen, usar `src/analysis/summary.py`.

## Autor

Marcos Aguilella\
IDIVAL\
marcos.aguilella@idival.org

## Citación

Si usas este código en tu investigación, por favor cita:

```
@MastersThesis{aguilella2026,
  author = {Marcos Aguilella Fabregat},
  title = {Voice Analysis using Artificial Intelligence for the early diagnosis of Parkinson's disease associated with **LRKK2** mutation},
  school = {IDIVAL},
  year = {2026}
}
```

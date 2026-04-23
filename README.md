# Voice Analysis for Parkinson's Disease Detection

Master's Thesis Code Repository - TFM_code

## Overview

This repository contains code for analyzing voice recordings to detect Parkinson's Disease (PD) and classify disease stages. The code compares Machine Learning (ML) and Deep Learning (DL) approaches.

## Key Features

- **Patient-level cross-validation**: Uses `StratifiedGroupKFold` to prevent data leakage — all audio clips from the same patient are in either train OR test, never both
- **Multiple feature extractors**: OpenSMILE, Praat (Parselmouth), Librosa
- **ML models**: Random Forest, SVM, XGBoost
- **DL models**: CNN, Bi-LSTM, ResNet, EfficientNet, Audio Spectrogram Transformer

## Directory Structure

```
TFM_code/
├── src/
│   ├── data/              # Data loading
│   │   └── humv_loader.py
│   │
│   ├── preprocessing/    # Audio preprocessing
│   │   ├── audio_processor.py
│   │   └── splitter.py      # Patient-level splitting
│   │
│   ├── features/       # Feature extraction
│   │   ├── opensmile.py
│   │   ├── praat.py
│   │   └── librosa_features.py
│   │
│   ├── models/       # Model definitions
│   │   ├── pytorch/   # DL models
│   │   └── sklearn/   # ML models
│   │
│   ├── training/   # Training pipelines
│   │   ├── cross_validation.py
│   │   ├── sklearn_trainer.py
│   │   └── pytorch_trainer.py
│   │
│   ├── dataset/    # PyTorch Datasets
│   │   └── audio_dataset.py
│   │
│   └── utils/     # Utilities
│
├── data/        # Data (gitignored)
└── outputs/    # Results (gitignored)
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Load Audio Data

```python
from src.data import load_audio_data

df = load_audio_data(
    root_directory='/path/to/audio/files',
    start_with='vocal'  # Filter files starting with 'vocal'
)

print(f"Loaded {len(df)} audio files")
print(df['Label'].value_counts())
```

### 2. Preprocess Audio

```python
from src.preprocessing import execute_preprocess_and_split

audio_chunks, labels, patient_ids = execute_preprocess_and_split(
    df,
    chunk_duration=5,      # 5-second chunks
    target_sr=16000,       # 16 kHz
    remove_silence=True
)

print(f"Generated {len(audio_chunks)} chunks")
```

### 3. Extract Features

```python
from src.features import extract_opensmile_features

features_df = extract_opensmile_features(
    audio_chunks,
    labels,
    patient_ids
)
```

### 4. Train ML Model (with patient-level CV)

```python
from src.models.sklearn import SVMClassifier
from src.training import SklearnTrainer
from src.preprocessing.splitter import get_patient_labels

# Get patient IDs and labels
patient_ids_unique, patient_labels = get_patient_labels(patient_ids, labels)

# Prepare features
X = features_df.drop(columns=['patient_id', 'label']).values
y = features_df['label'].values

# Train with CV
model = SVMClassifier(kernel='rbf', C=1.0)
trainer = SklearnTrainer(model=model, n_splits=5)

results = trainer.train(X, y, patient_ids)
```

### 5. Train DL Model

```python
from src.models.pytorch import CNN2D
from src.dataset import ParkinsonAudioDataset
from src.training import PyTorchTrainer
from torch.utils.data import DataLoader

dataset = ParkinsonAudioDataset(audio_chunks, labels, patient_ids)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CNN2D(num_classes=2)
trainer = PyTorchTrainer(model=model, n_epochs=50)

history = trainer.train(train_loader)
```

## Data Leakage Prevention

**CRITICAL**: This codebase uses patient-level splitting to prevent data leakage.

```python
from src.preprocessing.splitter import split_by_patients

for train_idx, test_idx in split_by_patients(patient_ids, labels, n_splits=5):
    train_patients = set(patient_ids[train_idx])
    test_patients = set(patient_ids[test_idx])
    
    # This will raise an assertion error if there's leakage
    assert train_patients.isdisjoint(test_patients), "LEAKAGE DETECTED!"
```

## Feature Extractors

| Feature Set | Description | Use Case |
|------------|-------------|----------|
| OpenSMILE | ComParE 2016, eGeMAPS | Standard acoustic features |
| Praat | Pitch, formants, jitter, shimmer, HNR | Phonatory features |
| Librosa | MFCCs, spectral contrast | Timbre features |

## Model Comparison

### ML Models (sklearn)

- **Random Forest**: Robust ensemble method
- **SVM**: Good for high-dimensional features
- **XGBoost**: Gradient boosting

### DL Models (PyTorch)

- **CNN1D**: Raw waveform classification
- **CNN2D**: Spectrogram classification
- **Bi-LSTM**: Temporal modeling
- **ResNet/EfficientNet**: Transfer learning from ImageNet
- **AST**: Audio Spectrogram Transformer

## Results

Patient-wise metrics are calculated by aggregating audio-level predictions:

```python
from src.training.metrics import calculate_patient_wise_metrics

patient_results, metrics = calculate_patient_wise_metrics(results_df)

print(f"Patient-wise AUC: {metrics['auc']:.3f}")
print(f"Patient-wise Accuracy: {metrics['accuracy']:.3f}")
```

## Common Workflows

### HC vs PD Classification

```python
# Filter to binary classification (HC=0 vs PD=1)
from src.data import load_audio_data, filter_binary

df = load_audio_data(root_directory)
df_binary = filter_binary(df, labels_to_keep=[0, 2])  # 0=HC, 2=PD -> 0, 1
```

### Multi-class (HC vs AC vs PD)

```python
# Keep all three classes
from src.data import load_audio_data

df = load_audio_data(root_directory)
# Labels: HC=0, NFC=0, AC=1, PD=2
```

## Data Directory Structure

```
root/
├── HC/
│   └── HUMV_HC_001/
│       └── vocal.wav
├── NFC/
├── AC/
│   └── HUMV_AC_001/
│       └── vocal.wav
└── PD/
    └── HUMV_PD_001/
        └── vocal.wav
```

## Requirements

See `requirements.txt`

## Author

Marcos Aguilella  
IDIVAL  
marcos.aguilella@idival.org

## Citation

If you use this code in your research, please cite:

```
@MastersThesis{aguilella2024,
  author = {Marcos Aguilella},
  title = {Voice Analysis for Parkinson's Disease Detection},
  school = {IDIVAL},
  year = {2024}
}
```
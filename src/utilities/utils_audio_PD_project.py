# -*- coding: utf-8 -*-
# @Time    : 21/1/25 11:38 AM
# @Author  : Marcos Aguilella
# @Affiliation  : IDIVAL
# @Email   : marcos.aguilella@idival.org
# @File    : utils_audio_PD_project.py


import pandas as pd
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.utils import resample
import os


def oversample_training_data(X_train, y_train, patient_ids, domain_labels, GRL):
  
	"""
	Oversample the minority class in the training data while keeping track of patient IDs.
	Args:
	X_train (np.ndarray): Training data features.
	y_train (np.ndarray): Training data labels.
	patient_ids (np.ndarray): Patient IDs for the training data.
	Returns:
	X_train_oversampled, y_train_oversampled, train_patient_ids_oversampled: Oversampled training data and patient IDs.
	"""
    
	# Combine X_train, y_train, and patient_ids into a single DataFrame for oversampling
	train_data = pd.DataFrame({
	"Feature": list(X_train),
	"Label": y_train,
	"Patient": patient_ids,
	"Domain": domain_labels if domain_labels is not None else None
	})

	# Separate majority and minority classes
	majority_class = train_data[train_data["Label"] == train_data["Label"].mode()[0]]
	minority_class = train_data[train_data["Label"] != train_data["Label"].mode()[0]]

	# Oversample minority class
	minority_oversampled = resample(
	minority_class,
	replace=True,	# Sample with replacement
	n_samples=len(majority_class),	# Match majority class size
	random_state=42
	)

	# Combine majority class and oversampled minority class
	oversampled_data = pd.concat([majority_class, minority_oversampled])

	# Shuffle the oversampled dataset to ensure random order
	oversampled_data = oversampled_data.sample(frac=1, random_state=42)

	# Split back into features, labels, and patient IDs
	X_train_oversampled = np.array(oversampled_data["Feature"].tolist())
	y_train_oversampled = oversampled_data["Label"].values
	train_patient_ids_oversampled = oversampled_data["Patient"].values
	domain_labels_oversampled = oversampled_data["Domain"].values if GRL else None

	return X_train_oversampled, y_train_oversampled, train_patient_ids_oversampled, domain_labels_oversampled

  
# Function to load and sort files properly
def load_processed_audio_dataset(directory, country_label, GRL, exercise_names=None):
    all_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    def exercise_filter(f, prefix):
        if exercise_names is None:
            return True
        names = exercise_names if isinstance(exercise_names, (list, tuple)) else [exercise_names]
        return any(f"{prefix}_5s_with_1s_overlap_{name}" in f for name in names)

    # Sort and filter files to maintain consistent order
    patient_id_files = sorted([f for f in all_files if exercise_filter(f, "patient_ids")])
    label_files = sorted([f for f in all_files if exercise_filter(f, "labels")])
    audio_files = sorted([f for f in all_files if exercise_filter(f, "audio_segments")])

    print(f"Loading dataset from: {directory}")
    print("Selected exercises:", exercise_names)
    print("Sorted Patient ID files:", patient_id_files)
    print("Sorted Label files:", label_files)
    print("Sorted Audio files:", audio_files)

    # Load data
    patient_ids = np.concatenate([np.load(os.path.join(directory, f)) for f in patient_id_files])
    labels = np.concatenate([np.load(os.path.join(directory, f)) for f in label_files])
    audio_segments = np.concatenate([np.load(os.path.join(directory, f)) for f in audio_files])

    # Assign domain label based on the dataset source
    if GRL:
        domain_labels = np.full(len(patient_ids), country_label)
    else:
        domain_labels = None

    return patient_ids, labels, audio_segments, domain_labels

class AverageMeter:
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def split_by_patients(patient_ids, labels, n_splits=5, shuffle=True, random_state=42):
    """
    Creates train/test splits ensuring all audio clips from the same patient are in the same split.

    This is CRITICAL to prevent data leakage - using StratifiedGroupKFold to ensure:
    1. No patient appears in both train and test sets
    2. Proportion of classes is preserved in each split

    Args:
        patient_ids (np.array): Array of patient IDs.
        labels (np.array): Array of labels.
        n_splits (int): Number of splits (for cross-validation).
        shuffle (bool): Whether to shuffle.
        random_state (int): Random seed for reproducibility.

    Yields:
        tuple: (train_idx, test_idx) for each fold.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    unique_patients = np.unique(patient_ids)
    patient_labels = np.array([
        labels[patient_ids == p][0] for p in unique_patients
    ])

    for fold, (train_patients, test_patients) in enumerate(
        sgkf.split(unique_patients, patient_labels, groups=unique_patients)
    ):
        train_patient_set = set(unique_patients[train_patients])
        test_patient_set = set(unique_patients[test_patients])

        assert train_patient_set.isdisjoint(test_patient_set), \
            f"LEAKAGE DETECTED in fold {fold}! Patients in both train and test!"

        train_idx = np.where(np.isin(patient_ids, train_patient_set))[0]
        test_idx = np.where(np.isin(patient_ids, test_patient_set))[0]

        yield train_idx, test_idx


def split_train_test(
    patient_ids,
    labels,
    test_size=0.2,
    shuffle=True,
    random_state=42
):
    """
    Creates a single train/test split with patient-level separation.

    Args:
        patient_ids (np.array): Array of patient IDs.
        labels (np.array): Array of labels.
        test_size (float): Proportion for test set.
        shuffle (bool): Whether to shuffle.
        random_state (int): Random seed.

    Returns:
        tuple: (train_idx, test_idx).
    """
    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=shuffle,
        random_state=random_state
    )

    unique_patients = np.unique(patient_ids)
    patient_labels = np.array([
        labels[patient_ids == p][0] for p in unique_patients
    ])

    train_idx, test_idx = next(sgkf.split(
        unique_patients, patient_labels, groups=unique_patients
    ))

    train_patient_set = set(unique_patients[train_idx])
    test_patient_set = set(unique_patients[test_idx])

    assert train_patient_set.isdisjoint(test_patient_set), \
        "LEAKAGE DETECTED! Patients in both train and test!"

    train_idx_full = np.where(np.isin(patient_ids, train_patient_set))[0]
    test_idx_full = np.where(np.isin(patient_ids, test_patient_set))[0]

    return train_idx_full, test_idx_full


def get_patient_labels(patient_ids, labels):
    """
    Get unique patient IDs with their corresponding labels.

    Args:
        patient_ids (np.array): Array of patient IDs.
        labels (np.array): Array of labels.

    Returns:
        tuple: (unique_patients, patient_labels)
    """
    patient_label_map = {}
    for pid, label in zip(patient_ids, labels):
        if pid not in patient_label_map:
            patient_label_map[pid] = label

    unique_patients = np.array(list(patient_label_map.keys()))
    patient_labels = np.array([patient_label_map[p] for p in unique_patients])

    return unique_patients, patient_labels
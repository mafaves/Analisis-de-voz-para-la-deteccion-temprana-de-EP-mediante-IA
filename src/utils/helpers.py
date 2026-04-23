import os
import numpy as np
import json


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_class_weights(labels):
    """Compute class weights for imbalanced datasets."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return weights


def oversample_minority(X, y, patient_ids=None, random_state=42):
    """
    Random oversampling of minority class.

    Args:
        X (np.array): Features.
        y (np.array): Labels.
        patient_ids (np.array, optional): Patient IDs (for consistency).
        random_state (int): Random seed.

    Returns:
        tuple: Resampled X, y, patient_ids.
    """
    np.random.seed(random_state)

    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    target_counts = {c: max_count for c in classes}

    X_resampled = [X[y == classes[0]]
    y_resampled = [y[y == classes[0]]]
    ids_resampled = [patient_ids[y == classes[0]]] if patient_ids is not None else [None]

    for cls in classes[1:]:
        cls_mask = y == cls
        X_cls = X[cls_mask]
        n_samples = max_count - len(X_cls)

        if n_samples > 0:
            idx = np.random.choice(len(X_cls), n_samples, replace=True)
            X_resampled.append(np.vstack([X_cls, X_cls[idx]]))
            y_resampled.append(np.hstack([y[cls_mask], y[cls_mask][idx]]))
            if patient_ids is not None:
                ids_resampled.append(np.hstack([patient_ids[cls_mask], patient_ids[cls_mask][idx]]))
        else:
            X_resampled.append(X_cls)
            y_resampled.append(y[cls_mask])
            if patient_ids is not None:
                ids_resampled.append(patient_ids[cls_mask])

    X_out = np.vstack(X_resampled)
    y_out = np.hstack(y_resampled)

    shuffle_idx = np.random.permutation(len(y_out))
    X_out = X_out[shuffle_idx]
    y_out = y_out[shuffle_idx]

    if patient_ids is not None:
        ids_out = np.hstack(ids_resampled)[shuffle_idx]
        return X_out, y_out, ids_out

    return X_out, y_out
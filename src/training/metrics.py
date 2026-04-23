import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)


def calculate_metrics(y_true, y_pred, y_proba=None, average='binary'):
    """
    Calculate classification metrics.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        y_proba (np.array, optional): Predicted probabilities.
        average (str): Averaging method for multiclass.

    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        metrics['specificity'] = None

    if y_proba is not None:
        if y_proba.ndim == 2 and y_proba.shape[1] > 1:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1], multi_class='ovr')
        else:
            metrics['auc'] = roc_auc_score(y_true, y_proba)

    return metrics


def calculate_patient_wise_metrics(results_df, num_classes=2):
    """
    Calculate patient-wise metrics from audio-level predictions.

    Aggregates audio predictions per patient (e.g., by averaging probabilities)
    and calculates metrics at the patient level.

    THIS IS THE CORRECT WAY TO REPORT FINAL RESULTS for medical diagnosis,
    since patients should be the unit of analysis, not individual audio clips.

    Args:
        results_df (pd.DataFrame): DataFrame with columns:
            - 'patient_id': Patient ID
            - 'label': True label (0 or 1)
            - 'pred_value': Predicted probability
        num_classes (int): Number of classes (2 for binary).

    Returns:
        tuple: (patient_results_df, metrics_dict)
    """
    patient_results = (
        results_df.groupby('patient_id')
        .agg({
            'label': 'first',
            'pred_value': lambda x: np.mean([p for p in x], axis=0)
        })
        .reset_index()
    )

    if num_classes == 2:
        patient_results['pred'] = patient_results['pred_value'].apply(
            lambda x: 1 if x[1] > 0.5 else 0
        )
    else:
        patient_results['pred'] = patient_results['pred_value'].apply(
            lambda x: np.argmax(x)
        )

    y_true = patient_results['label'].values
    y_pred = patient_results['pred'].values
    y_proba = np.array(patient_results['pred_value'].tolist())

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba[:, 1]) if num_classes == 2 else roc_auc_score(y_true, y_proba, multi_class='ovr'),
        'specificity': specificity,
        'confusion_matrix': cm.tolist(),
        'n_patients': len(patient_results),
        'n_patients_per_class': {
            'class_0': int((y_true == 0).sum()),
            'class_1': int((y_true == 1).sum())
        }
    }

    return patient_results, metrics


def get_classification_report(y_true, y_pred):
    """Get detailed classification report."""
    return classification_report(y_true, y_pred, zero_division=0)
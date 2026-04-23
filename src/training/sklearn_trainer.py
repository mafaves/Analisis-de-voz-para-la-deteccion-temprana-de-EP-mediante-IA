import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import json
import os


class SklearnTrainer:
    """
    Trainer for sklearn ML models with cross-validation.

    Handles:
    - Patient-level train/test splitting (prevents data leakage)
    - Hyperparameter tuning with GridSearchCV
    - Model evaluation and metrics

    Example:
        >>> from src.models.sklearn import SVMClassifier
        >>> from src.training import SklearnTrainer
        >>> trainer = SklearnTrainer(model=SVMClassifier())
        >>> results = trainer.train(X, y, patient_ids)
    """

    def __init__(
        self,
        model,
        param_grid=None,
        n_splits=5,
        inner_cv=3,
        scoring='accuracy',
        scale_features=True,
        random_state=42
    ):
        self.model = model
        self.param_grid = param_grid or {}
        self.n_splits = n_splits
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.scale_features = scale_features
        self.random_state = random_state

        self.scaler_ = StandardScaler()
        self.best_model_ = None
        self.results_ = None

    def train(
        self,
        X,
        y,
        patient_ids,
        save_dir=None,
        verbose=True
    ):
        """
        Train model with nested cross-validation.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Labels.
            patient_ids (np.array): Patient IDs for group-level splitting.
            save_dir (str, optional): Directory to save results.
            verbose (bool): Print progress.

        Returns:
            dict: Training results and metrics.
        """
        from .cross_validation import StratifiedGroupKFold

        cv = StratifiedGroupKFold(n_splits=self.n_splits, random_state=self.random_state)

        fold_results = []
        all_metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'auc': [], 'specificity': []
        }

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, patient_ids)):
            if verbose:
                print(f"\n=== Fold {fold + 1}/{self.n_splits} ===")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_patients = patient_ids[train_idx]
            test_patients = patient_ids[test_idx]

            assert set(train_patients).isdisjoint(set(test_patients)), \
                "DATA LEAKAGE DETECTED!"

            if self.scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if self.param_grid:
                inner_cv = StratifiedGroupKFold(
                    n_splits=self.inner_cv,
                    random_state=self.random_state + fold
                )
                grid = GridSearchCV(
                    self.model,
                    self.param_grid,
                    cv=inner_cv.split(X_train, y_train, patient_ids[train_idx]),
                    scoring=self.scoring,
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                self.model.fit(X_train, y_train)
                model = self.model
                best_params = {}

            y_pred = model.predict(X_test)

            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob),
                'specificity': specificity,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'best_params': best_params
            }

            # for k, v in metrics.items():
            #     if k not in ['best_params']:
            #         all_metrics[k].append(v)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

            fold_results.append(metrics)

            if verbose:
                print(f"  Accuracy: {metrics['accuracy']:.3f}")
                print(f"  AUC: {metrics['auc']:.3f}")
                print(f"  Sensitivity: {metrics['recall']:.3f}")
                print(f"  Specificity: {metrics['specificity']:.3f}")

        self.results_ = {
            'fold_results': fold_results,
            'mean_metrics': {k: np.mean(v) for k, v in all_metrics.items()},
            'std_metrics': {k: np.std(v) for k, v in all_metrics.items()}
        }

        if save_dir:
            self.save_results(save_dir)

        return self.results_

    def predict(self, X):
        """Predict using best model."""
        if self.best_model_ is None:
            raise ValueError("Model not trained yet!")
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using best model."""
        if self.best_model_ is None:
            raise ValueError("Model not trained yet!")
        return self.best_model_.predict_proba(X)

    def save_results(self, save_dir):
        """Save training results to directory."""
        os.makedirs(save_dir, exist_ok=True)

        df = pd.DataFrame(self.results_['fold_results'])
        df.to_csv(os.path.join(save_dir, 'fold_results.csv'), index=False)

        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump({
                'mean': self.results_['mean_metrics'],
                'std': self.results_['std_metrics']
            }, f, indent=2)

        print(f"Results saved to {save_dir}")
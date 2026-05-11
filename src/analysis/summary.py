import os
import json
import pandas as pd


def generate_summary_csv(
    results_dir,
    output_file='summary_patient_level.csv',
    metrics_to_include=None
):
    """
    Scans experiment results and creates a CSV summary at patient level.

    Expected structure:
        {results_dir}/
        ├── exercise_1/
        │   ├── feature_type_1/metrics.json
        │   └── feature_type_2/metrics.json
        └── exercise_2/
            └── ...

    Each metrics.json must contain:
    {
        "exercise": "...",
        "feature_type": "...",
        "patient_level": {
            "mean": {"accuracy": ..., "recall": ..., ...},
            "std": {"accuracy": ..., "recall": ..., ...}
        }
    }

    Args:
        results_dir (str): Base directory with experiment results.
        output_file (str): Output CSV filename (saved in results_dir).
        metrics_to_include (list, optional): Metrics to include.
            Default: ['accuracy', 'recall', 'f1', 'specificity', 'auc']
    """
    if metrics_to_include is None:
        metrics_to_include = ['accuracy', 'recall', 'f1', 'specificity', 'auc']

    rows = []

    for exercise in sorted(os.listdir(results_dir)):
        exercise_dir = os.path.join(results_dir, exercise)
        if not os.path.isdir(exercise_dir):
            continue

        for feature_type in sorted(os.listdir(exercise_dir)):
            feature_dir = os.path.join(exercise_dir, feature_type)
            metrics_path = os.path.join(feature_dir, 'metrics.json')

            if not os.path.isdir(feature_dir) or not os.path.isfile(metrics_path):
                continue

            with open(metrics_path, 'r') as f:
                data = json.load(f)

            patient = data.get('patient_level', {})
            mean = patient.get('mean', {})
            std = patient.get('std', {})

            row = {
                'exercise': data.get('exercise', exercise),
                'feature_type': data.get('feature_type', feature_type),
                'n_folds': data.get('n_folds', '')
            }

            for metric in metrics_to_include:
                m = mean.get(metric)
                s = std.get(metric)
                if m is not None and s is not None:
                    row[metric] = f"{m:.3f} ± {s:.3f}"
                elif m is not None:
                    row[metric] = f"{m:.3f}"
                else:
                    row[metric] = ''

            rows.append(row)

    if not rows:
        print("No metrics.json files found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    columns = ['exercise', 'feature_type', 'n_folds'] + metrics_to_include
    df = df[[c for c in columns if c in df.columns]]

    output_path = os.path.join(results_dir, output_file)
    df.to_csv(output_path, index=False)

    print(f"\nSummary saved to: {output_path}")
    print(f"Found {len(df)} experiment configurations:\n")
    print(df.to_string(index=False))

    return df
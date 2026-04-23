import pandas as pd
import numpy as np
import argparse
import os
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from collections import defaultdict
from statistics import mode as py_mode

def parse_label_map(label_map_str):
    return {int(k): int(v) for k, v in (pair.split(":") for pair in label_map_str.split(","))}

def compute_specificity(cm):
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def write_summary(f, title, scores):
    f.write(f"{title}:\n")
    valid_metric_means = []
    for k, vals in scores.items():
        valid_vals = [v for v in vals if v is not None]
        if len(valid_vals) == 0:
            mean, std = float('nan'), float('nan')
        else:
            mean, std = float(np.mean(valid_vals)), float(np.std(valid_vals))
            valid_metric_means.append(mean)
        f.write(f"{k:<10}: {mean:.3f} ({std:.3f})\n")
    if len(valid_metric_means) > 0:
        final_score = sum(valid_metric_means) / len(valid_metric_means)
        f.write(f"Final score: {final_score:.3f}\n")
    else:
        f.write("Final score: nan\n")
    f.write("\n")

def main(file_path, feature_type, exercise, label_map, save_dir):
    
    #print(f"Loading data from: {file_path}")
    # Load and preprocess
    df = pd.read_csv(file_path, sep=',', index_col=None, decimal='.')

    # df['age'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')
    # df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')

    df['label'] = df['label'].astype(int)
    df['age'] = df['age'].astype(int)

    # Filter by exercise and relabel
    df = df[df['exercise'] == exercise]
    df = df[df['label'].isin(label_map.keys())].copy()
    df['label'] = df['label'].map(label_map)
    print(f"Data shape after filtering: {df.shape}")

    # Assume ac_patient_ids is a list of patient IDs for AC group
    ac_patient_ids = ["HUMV_AC_1", "HUMV_AC_3", "HUMV_AC_4", "HUMV_AC_6", "HUMV_AC_9", "HUMV_AC_10",
    "HUMV_AC_11", "HUMV_AC_12", "HUMV_AC_17", "HUMV_AC_18", "HUMV_AC_19",
    "HUMV_AC_20", "HUMV_AC_21", "HUMV_AC_22", "HUMV_AC_23",
    "HUMV_AC_25", "HUMV_AC_30", "HUMV_AC_33", "HUMV_AC_34",
    "HUMV_AC_24", "HUMV_AC_32"] 

    # Drop NaNs and metadata
    # # Remove AC patients from main data
    df_HC_vs_PD = df[~df['id'].isin(ac_patient_ids)]
    print(f"Data shape after removing AC patients: {df_HC_vs_PD.shape}")   
    
    X = df_HC_vs_PD.drop(columns=['id', 'label', 'exercise', 'age', 'sex',  'segment'])
    y = df_HC_vs_PD['label']
    groups = df_HC_vs_PD['id']
    print(f"Unique patients after removing AC: {groups.nunique()}")
    nan_cols = X.isnull().sum()
    columns_removed = nan_cols[nan_cols > 0].index.to_list()
    X = X.drop(columns=columns_removed)


    # Separate AC data
    df_ac = df[df['id'].isin(ac_patient_ids)].copy()
    X_ac = df_ac.drop(columns=['id', 'label', 'exercise', 'age', 'sex', 'segment'])
    X_ac = X_ac.drop(columns=[col for col in columns_removed if col in X_ac.columns])
    y_ac = df_ac['label']
    groups_ac = df_ac['id']

    # Create a mapping of patient IDs to their diagnosis (PD = 1, HC = 0)
    patient_label_map = {}
    for patient_id, label in zip(groups, y):
        if patient_id not in patient_label_map:
            patient_label_map[patient_id] = label  # Assign the first encountered label (assume consistent labels)

    # Extract unique patients and their corresponding labels
    unique_patients = np.array(list(patient_label_map.keys()))  # Unique patient IDs
    patient_labels = np.array([patient_label_map[pid] for pid in unique_patients])  # PD = 1, HC = 0
    
    print(f"Total unique patients: {len(unique_patients)}")
    print(f"Patients per class: {np.bincount(patient_labels)}")  # Count of patients in each class


    print(f"\n\n === Using exercise: {exercise} === \n\n")
    print(f"Relabeled using: {label_map}\n\n")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # outer_cv = GroupKFold(n_splits=5, random_state=42, shuffle=True)
    outer_cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    # outer_cv= StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    param_grids = {
        'linear': {'C': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
        'sigmoid': {'C': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100], 'kernel': ['sigmoid'], 'gamma': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]},
        'rbf': {'C': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]},
        'poly': {'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'gamma': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10]}
    }

    outer_results = []
    outer_results_ac = []

    for kernel_name, param_grid in param_grids.items():
        print(f"\n==== Running experiments for kernel: {kernel_name} ====")
        seen_patients = []

        #for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups), 1):
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(unique_patients, patient_labels, groups= unique_patients), 1): #
            train_patients = unique_patients[train_idx]
            test_patients = unique_patients[test_idx]

            # Count PD & HC in the training set
            train_PD_count = sum(patient_labels[train_idx] == 1)
            train_HC_count = sum(patient_labels[train_idx] == 0)

            # print(f"Fold {fold}: Final Train PD={train_PD_count}, Train HC={train_HC_count}")
            # print(f"Fold {fold}: Test PD={sum(patient_label_map[p] for p in test_patients)}, Test HC={len(test_patients) - sum(patient_label_map[p] for p in test_patients)}")

            train_indices = np.array([i for i, pid in enumerate(groups) if pid in train_patients])
            test_indices = np.array([i for i, pid in enumerate(groups) if pid in test_patients])

            X_train_raw, X_test_raw = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            groups_train, groups_test = groups.iloc[train_indices], groups.iloc[test_indices]

            # Get unique patients and their labels in the training set
            unique_train_patients = np.array(list({pid: label for pid, label in zip(groups_train, y_train)}.keys()))
            train_patient_labels = np.array([patient_label_map[pid] for pid in unique_train_patients])


            # X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
            # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

            assert not set(groups_train).intersection(groups_test), "Patient leakage detected!"
            seen_patients.append(set(groups_test))
            for past in seen_patients[:-1]:
                assert set(groups_test).isdisjoint(past), "Overlap between test sets!"

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            # Inner cv
            model = SVC(probability=False, class_weight = 'balanced')
            # inner_cv = GroupKFold(n_splits=3)
            # grid = GridSearchCV(model, param_grid, cv=inner_cv.split(X_train, y_train, groups_train), scoring='accuracy', n_jobs = 4) # data leakage because of X_train and y_train already scaled
            # grid.fit(X_train, y_train)

            # Use patient-level stratified group k-fold for inner CV
            inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=fold)  # random_state for reproducibility
            # Split at the patient level, then map back to segment indices
            inner_cv_splits = []
            for inner_train_idx, inner_val_idx in inner_cv.split(unique_train_patients, train_patient_labels, groups=unique_train_patients):
                inner_train_pids = unique_train_patients[inner_train_idx]
                inner_val_pids = unique_train_patients[inner_val_idx]
                inner_train_indices = np.where(groups_train.isin(inner_train_pids))[0]
                inner_val_indices = np.where(groups_train.isin(inner_val_pids))[0]
                inner_cv_splits.append((inner_train_indices, inner_val_indices))

                # Count PD & HC in inner train/val
                inner_train_labels = [patient_label_map[pid] for pid in inner_train_pids]
                inner_val_labels = [patient_label_map[pid] for pid in inner_val_pids]
                print(f"  Inner CV split: Train PD={inner_train_labels.count(1)}, Train HC={inner_train_labels.count(0)} | "
                    f"Val PD={inner_val_labels.count(1)}, Val HC={inner_val_labels.count(0)}")
                print(f"    X_train shape: {X_train[inner_train_indices].shape}, y_train shape: {y_train.iloc[inner_train_indices].shape}")
                print(f"    X_val shape: {X_train[inner_val_indices].shape}, y_val shape: {y_train.iloc[inner_val_indices].shape}")

            grid = GridSearchCV(
                model,
                param_grid,
                cv=inner_cv_splits,
                scoring='accuracy',
                n_jobs=4
            )
            grid.fit(X_train, y_train)

            # Train new model with best params
            best_params = grid.best_params_
            print(f"Fold {fold} best params: {best_params}, Final Train PD={train_PD_count}, Train HC={train_HC_count} ")

            best_model = SVC(**best_params, class_weight='balanced')
            best_model.fit(X_train, y_train)

            ##########
            # Segment-wise prediction 
            ##########

            y_pred = best_model.predict(X_test)
            decision_scores = best_model.decision_function(X_test)

            df_preds = pd.DataFrame({
                'decision_score': decision_scores,
                'id': groups_test.values,
                'true_label': y_test.values
            })

            try:
                segment_auc = roc_auc_score(y_test, decision_scores)
            except:
                segment_auc = None

            seg_cm = confusion_matrix(y_test, y_pred)
            seg_specificity = compute_specificity(seg_cm) if seg_cm.shape == (2, 2) else None

            ##########
            # Patient-wise prediction 
            ##########

            patient_scores = df_preds.groupby('id')['decision_score'].mean()
            patient_true = df_preds.groupby('id')['true_label'].first()
            patient_pred = (patient_scores > 0).astype(int)
            patient_true = patient_true.loc[patient_pred.index]

            try:
                patient_auc = roc_auc_score(patient_true, patient_scores)
            except:
                patient_auc = None

            pat_cm = confusion_matrix(patient_true, patient_pred)
            pat_specificity = compute_specificity(pat_cm) if pat_cm.shape == (2, 2) else None

            ##########
            # AC predictions
            ##########            

            if len(X_ac) > 0:
                X_ac_scaled = scaler.transform(X_ac)
                y_ac_pred = best_model.predict(X_ac_scaled)
                ac_decision_scores = best_model.decision_function(X_ac_scaled)

                df_ac_preds = pd.DataFrame({
                    'decision_score': ac_decision_scores,
                    'id': groups_ac.values,
                    'true_label': y_ac.values
                })

                try:
                    ac_segment_auc = roc_auc_score(y_ac, ac_decision_scores)
                except:
                    ac_segment_auc = None

                ac_seg_cm = confusion_matrix(y_ac, y_ac_pred)
                ac_seg_specificity = compute_specificity(ac_seg_cm) if ac_seg_cm.shape == (2, 2) else None

                # Patient-wise for AC
                ac_patient_scores = df_ac_preds.groupby('id')['decision_score'].mean()
                ac_patient_true = df_ac_preds.groupby('id')['true_label'].first()
                ac_patient_pred = (ac_patient_scores > 0).astype(int)
                ac_patient_true = ac_patient_true.loc[ac_patient_pred.index]

                try:
                    ac_patient_auc = roc_auc_score(ac_patient_true, ac_patient_scores)
                except:
                    ac_patient_auc = None

                ac_pat_cm = confusion_matrix(ac_patient_true, ac_patient_pred)
                ac_pat_specificity = compute_specificity(ac_pat_cm) if ac_pat_cm.shape == (2, 2) else None

                df_ac_patients = pd.DataFrame({
                'id': ac_patient_scores.index,
                'true_label': ac_patient_true.values,
                'pred_label': ac_patient_pred.values,
                'decision_score': ac_patient_scores.values
                    })
                df_ac_patients['correct'] = df_ac_patients['true_label'] == df_ac_patients['pred_label']

                preds_file = os.path.join(
                save_dir,
                exercise,
                feature_type,
                f"ac_predictions_{exercise}_{kernel_name}_fold{fold}.csv"
                )
                if not os.path.exists(os.path.dirname(preds_file)):
                    os.makedirs(os.path.dirname(preds_file)) 
                #df_ac_preds.to_csv(preds_file, index=False, sep="\t")
                df_ac_patients.to_csv(preds_file, index=False, sep="\t")



            ##########
            # Results
            ##########

            ###
            # HC vs PD
            ###

            outer_results.append({
                'kernel': kernel_name,
                'fold': fold,
                'best_params': best_params,
                'segment_level': {
                    'y_true': y_test.values,
                    'y_pred': y_pred,
                    'confusion_matrix': seg_cm.tolist(),
                    'metrics': {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'roc_auc': segment_auc,
                        #'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
                        'specificity': seg_specificity
                    }
                },
                'patient_level': {
                    'y_true': patient_true.tolist(),
                    'y_pred': patient_pred.tolist(),
                    'confusion_matrix': pat_cm.tolist(),
                    'metrics': {
                        'accuracy': accuracy_score(patient_true, patient_pred),
                        'roc_auc': patient_auc,
                        #'precision': precision_score(patient_true, patient_pred, average='macro', zero_division=0),
                        'recall': recall_score(patient_true, patient_pred, average='macro', zero_division=0),
                        'f1': f1_score(patient_true, patient_pred, average='macro', zero_division=0),
                        'specificity': pat_specificity
                    }
                }
            })

            ###
            # AC
            ###

            # Store AC results
            outer_results_ac.append({
                'kernel': kernel_name,
                'fold': fold,
                'segment_level': {
                    'y_true': y_ac.values,
                    'y_pred': y_ac_pred,
                    'confusion_matrix': ac_seg_cm.tolist(),
                    'metrics': {
                        'accuracy': accuracy_score(y_ac, y_ac_pred),
                        'roc_auc': ac_segment_auc,
                        'recall': recall_score(y_ac, y_ac_pred, average='macro', zero_division=0),
                        'f1': f1_score(y_ac, y_ac_pred, average='macro', zero_division=0),
                        'specificity': ac_seg_specificity
                    }
                },
                'patient_level': {
                    'y_true': ac_patient_true.tolist(),
                    'y_pred': ac_patient_pred.tolist(),
                    'confusion_matrix': ac_pat_cm.tolist(),
                    'metrics': {
                        'accuracy': accuracy_score(ac_patient_true, ac_patient_pred),
                        'roc_auc': ac_patient_auc,
                        'recall': recall_score(ac_patient_true, ac_patient_pred, average='macro', zero_division=0),
                        'f1': f1_score(ac_patient_true, ac_patient_pred, average='macro', zero_division=0),
                        'specificity': ac_pat_specificity
                    }
                }
            })



    ##########
    # Save results
    ##########

    ###
    # HC vs PD
    ###

    results_by_kernel = defaultdict(list)
    for res in outer_results:
        results_by_kernel[res['kernel']].append(res)

    os.makedirs(os.path.join(save_dir, exercise, feature_type), exist_ok=True)
    
    summary_rows = []
    for kernel, results in results_by_kernel.items():
        segment_scores = defaultdict(list)
        patient_scores = defaultdict(list)
        result_file = os.path.join(save_dir, exercise, feature_type, f"results_{exercise}_{kernel}.txt")

        with open(result_file, "w") as f:
            if columns_removed:
                f.write(f"Columns with NaN (and removed): {columns_removed}\n\n")
            else:
                f.write("There are no missing values in any column.\n\n")

            for fold_result in results:
                f.write(f"--- Fold {fold_result['fold']} ---\n")
                f.write(f"Best Parameters: {fold_result['best_params']}\n\n")

                # Segment-level
                seg = fold_result['segment_level']
                f.write("Segment-Level Confusion Matrix:\n")
                f.write(str(seg['confusion_matrix']) + "\n\n")
                for k, v in seg['metrics'].items():
                    segment_scores[k].append(v)

                # Patient-level
                pat = fold_result['patient_level']
                f.write("Patient-Level Confusion Matrix:\n")
                f.write(str(pat['confusion_matrix']) + "\n\n")
                for k, v in pat['metrics'].items():
                    patient_scores[k].append(v)

                f.write("=" * 60 + "\n\n")

            write_summary(f, "Segment-Level Metrics", segment_scores)
            write_summary(f, "Patient-Level Metrics", patient_scores)
    
        # Collect metrics across folds (patient-level to match your table)
        metrics_collect = defaultdict(list)
        for fold in results:
            for k, v in fold['patient_level']['metrics'].items():
                if v is not None:
                    metrics_collect[k].append(v)

        # Compute mean ± std for each metric
        metrics_summary = {}
        for k, vals in metrics_collect.items():
            mean = np.mean(vals)
            std = np.std(vals)
            metrics_summary[k] = f"{mean:.3f} ({std:.3f})"
        
        # Store also numeric accuracy + final score for ranking
        acc_mean = np.mean(metrics_collect['accuracy']) if 'accuracy' in metrics_collect else 0
        final_score = np.mean([np.mean(v) for v in metrics_collect.values()]) if metrics_collect else -np.inf

        # Collect hyperparameters (mode voting)
        C_vals = [fold['best_params'].get('C', None) for fold in results]
        gamma_vals = [fold['best_params'].get('gamma', None) for fold in results]
        degree_vals = [fold['best_params'].get('degree', None) for fold in results]

        C_mode = py_mode([c for c in C_vals if c is not None]) if any(C_vals) else None
        gamma_mode = py_mode([g for g in gamma_vals if g is not None]) if any(gamma_vals) else None
        degree_mode = py_mode([d for d in degree_vals if d is not None]) if any(degree_vals) else None

        summary_rows.append({
            "Exercise": exercise,
            "Dimension": feature_type,
            "Kernel": kernel,
            "C": C_mode,
            "Gamma": gamma_mode,
            "Degree": degree_mode,
            "Accuracy": metrics_summary.get("accuracy", "nan"),
            "Recall": metrics_summary.get("recall", "nan"),
            "Specificity": metrics_summary.get("specificity", "nan"),
            "F1-score": metrics_summary.get("f1", "nan"),
            "AUC": metrics_summary.get("roc_auc", "nan"),
            "Accuracy_mean": acc_mean,   # numeric for ranking
            "Final_score": f"{final_score:.3f}"
        })

    # Pick best kernel per feature based on accuracy → then final score
    best_kernel_row = max(summary_rows, key=lambda x: (x["Accuracy_mean"], x["Final_score"]))
    best_kernel_row.pop("Accuracy_mean")  # drop helper column
    best_kernel_name = best_kernel_row["Kernel"]

    # Save into DataFrame
    summary_df = pd.DataFrame([best_kernel_row])
    os.makedirs(os.path.join(save_dir, exercise, feature_type), exist_ok=True)
    summary_df.to_csv(
        os.path.join(save_dir, exercise, feature_type, f"best_summary_{exercise}_{feature_type}.csv"),
        index=False, sep = "\t")
    

    ###
    # AC
    ###

    # results_ac_by_kernel = defaultdict(list)
    # for res in outer_results_ac:
    #     results_ac_by_kernel[res['kernel']].append(res)


    # ac_summary_rows = []
    # for kernel, results in results_ac_by_kernel.items():
    #     ac_segment_scores = defaultdict(list)
    #     ac_patient_scores = defaultdict(list)
    #     ac_result_file = os.path.join(save_dir, exercise, feature_type, f"results_AC_{exercise}_{kernel}.txt")

    #     with open(ac_result_file, "w") as f:
    #         for fold_result in results:
    #             f.write(f"--- Fold {fold_result['fold']} ---\n")

    #             # Segment-level
    #             seg = fold_result['segment_level']
    #             f.write("Segment-Level Confusion Matrix:\n")
    #             f.write(str(seg['confusion_matrix']) + "\n\n")
    #             for k, v in seg['metrics'].items():
    #                 ac_segment_scores[k].append(v)

    #             # Patient-level
    #             pat = fold_result['patient_level']
    #             f.write("Patient-Level Confusion Matrix:\n")
    #             f.write(str(pat['confusion_matrix']) + "\n\n")
    #             for k, v in pat['metrics'].items():
    #                 ac_patient_scores[k].append(v)

    #             f.write("=" * 60 + "\n\n")

    #         write_summary(f, "Segment-Level Metrics (AC)", ac_segment_scores)
    #         write_summary(f, "Patient-Level Metrics (AC)", ac_patient_scores)

    #     # Collect metrics across folds (patient-level)
    #     ac_metrics_collect = defaultdict(list)
    #     for fold in results:
    #         for k, v in fold['patient_level']['metrics'].items():
    #             if v is not None:
    #                 ac_metrics_collect[k].append(v)

    #     # Compute mean ± std for each metric
    #     ac_metrics_summary = {}
    #     for k, vals in ac_metrics_collect.items():
    #         mean = np.mean(vals)
    #         std = np.std(vals)
    #         ac_metrics_summary[k] = f"{mean:.3f} ({std:.3f})"
        
    #     # Store also numeric accuracy + final score for ranking
    #     acc_mean = np.mean(ac_metrics_collect['accuracy']) if 'accuracy' in ac_metrics_collect else 0
    #     final_score = np.mean([np.mean(v) for v in ac_metrics_collect.values()]) if ac_metrics_collect else -np.inf
        

    #     ac_summary_rows.append({
    #         "Exercise": exercise,
    #         "Dimension": feature_type,
    #         "Kernel": kernel,
    #         "Accuracy": ac_metrics_summary.get("accuracy", "nan"),
    #         "Recall": ac_metrics_summary.get("recall", "nan"),
    #         "Specificity": ac_metrics_summary.get("specificity", "nan"),
    #         "F1-score": ac_metrics_summary.get("f1", "nan"),
    #         "AUC": ac_metrics_summary.get("roc_auc", "nan"),
    #         "Accuracy_mean": acc_mean,
    #         "Final_score": f"{final_score:.3f}"
    #     })
    
    # # Pick best kernel per feature based on accuracy → then final score
    # best_kernel_row_AC = max(ac_summary_rows, key=lambda x: (x["Accuracy_mean"], x["Final_score"]))
    # best_kernel_row_AC.pop("Accuracy_mean")  # drop helper column

    # # Save into DataFrame
    # summary_df_AC = pd.DataFrame([best_kernel_row_AC])
    # os.makedirs(os.path.join(save_dir, exercise, feature_type), exist_ok=True)
    # summary_df_AC.to_csv(
    #     os.path.join(save_dir, exercise, feature_type, f"best_AC_summary_{exercise}_{feature_type}.csv"),
    #     index=False, sep = "\t")
            

### --- AC summary: only save the row for the best kernel from HC vs PD --- ###
    results_ac_by_kernel = defaultdict(list)
    for res in outer_results_ac:
        results_ac_by_kernel[res['kernel']].append(res)

    ac_summary_rows = []
    for kernel, results in results_ac_by_kernel.items():
        ac_segment_scores = defaultdict(list)
        ac_patient_scores = defaultdict(list)
        ac_result_file = os.path.join(save_dir, exercise, feature_type, f"results_AC_{exercise}_{kernel}.txt")

        with open(ac_result_file, "w") as f:
            for fold_result in results:
                f.write(f"--- Fold {fold_result['fold']} ---\n")

                # Segment-level
                seg = fold_result['segment_level']
                f.write("Segment-Level Confusion Matrix:\n")
                f.write(str(seg['confusion_matrix']) + "\n\n")
                for k, v in seg['metrics'].items():
                    ac_segment_scores[k].append(v)

                # Patient-level
                pat = fold_result['patient_level']
                f.write("Patient-Level Confusion Matrix:\n")
                f.write(str(pat['confusion_matrix']) + "\n\n")
                for k, v in pat['metrics'].items():
                    ac_patient_scores[k].append(v)

                f.write("=" * 60 + "\n\n")

            write_summary(f, "Segment-Level Metrics (AC)", ac_segment_scores)
            write_summary(f, "Patient-Level Metrics (AC)", ac_patient_scores)

        # Collect metrics across folds (patient-level)
        ac_metrics_collect = defaultdict(list)
        for fold in results:
            for k, v in fold['patient_level']['metrics'].items():
                if v is not None:
                    ac_metrics_collect[k].append(v)

        # Compute mean ± std for each metric
        ac_metrics_summary = {}
        for k, vals in ac_metrics_collect.items():
            mean = np.mean(vals)
            std = np.std(vals)
            ac_metrics_summary[k] = f"{mean:.3f} ({std:.3f})"
        
        # Store also numeric accuracy + final score for ranking
        acc_mean = np.mean(ac_metrics_collect['accuracy']) if 'accuracy' in ac_metrics_collect else 0
        final_score = np.mean([np.mean(v) for v in ac_metrics_collect.values()]) if ac_metrics_collect else -np.inf
        

        ac_summary_rows.append({
            "Exercise": exercise,
            "Dimension": feature_type,
            "Kernel": kernel,
            "Accuracy": ac_metrics_summary.get("accuracy", "nan"),
            "Recall": ac_metrics_summary.get("recall", "nan"),
            "Specificity": ac_metrics_summary.get("specificity", "nan"),
            "F1-score": ac_metrics_summary.get("f1", "nan"),
            "AUC": ac_metrics_summary.get("roc_auc", "nan"),
            "Accuracy_mean": acc_mean,
            "Final_score": f"{final_score:.3f}"
        })

    # --- Only save the AC summary for the best kernel from HC vs PD ---
    best_ac_row = next(row for row in ac_summary_rows if row["Kernel"] == best_kernel_name)
    best_ac_row.pop("Accuracy_mean")  # drop helper column

    summary_df_AC = pd.DataFrame([best_ac_row])
    os.makedirs(os.path.join(save_dir, exercise, feature_type), exist_ok=True)
    summary_df_AC.to_csv(
        os.path.join(save_dir, exercise, feature_type, f"best_AC_summary_{exercise}_{feature_type}.csv"),
        index=False, sep = "\t")
    # ...existing code...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SVM experiments with custom settings.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--feature_type', type = str, required = True, help = 'Type of feature studied (prosody, articulation, joined...)')
    parser.add_argument('--exercise', type=str, required=True, help='Exercise to filter on')
    parser.add_argument('--label_map', type=str, required=True, help='Mapping of original labels to binary, e.g. "0:0,3:1"')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')

    args = parser.parse_args()
    label_map = parse_label_map(args.label_map)

    main(args.file_path, args.feature_type, args.exercise, label_map, args.save_dir)


#Usage
# cd Documents/IDIVAL/Proyectos/Audio_Parkinson/Colaboracion_grupo_Rafael_Orozco 
# python run_svm_pipeline.py --file_path "./files/Joined_features/features_articulation_5s_with_1s_overlap.csv" --feature_type "articulation" --exercise "patachaka" --label_map "0:0, 3:1" --save_dir "./resultados_disvoice/HC_vs_PD/"

# files:
# "./files/Joined_features/features_joined_5s_with_1s_overlap.csv"
# "./files/Joined_features/features_articulation_5s_with_1s_overlap.csv"
# "./files/Joined_features/features_phonological_5s_with_1s_overlap.csv"
# "./files/Joined_features/features_prosody_5s_with_1s_overlap.csv"
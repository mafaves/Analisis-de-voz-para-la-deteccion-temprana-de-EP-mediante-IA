# -*- coding: utf-8 -*-
# @Time	: 21/1/25 10:48 AM
# @Author  : Marcos Aguilella
# @Affiliation  : IDIVAL
# @Email   : marcos.aguilella@idival.org
# @File	: stats.py


from sklearn import metrics
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

def calculate_stats(output, target):
	"""
	Calculate statistics including mAP, AUC, confusion matrix, etc.

	Args:
	  output: 2D tensor (samples_num, classes_num) or (samples_num, 1)
	  target: 2D tensor (samples_num, classes_num) or 1D tensor (samples_num,)

	Returns:
	  stats: list of statistics for each class/audio sample.
	  audio_results: per-sample probabilities, predictions, and true labels.
	"""

	# Ensure output and target are tensors
	if isinstance(output, list):
		output = torch.cat(output)  # Convert list of tensors into a single tensor
	if isinstance(target, list):
		target = torch.cat(target)

	# Ensure tensors are on CPU and converted to numpy
	output = output.detach().cpu()
	target = target.detach().cpu()

	# Determine the classification type
	if output.ndimension() == 1 or output.shape[1] == 1:
		# Binary classification
		output = output.squeeze()  # Remove singleton dimensions
		target = target.squeeze()  # Ensure target is 1D
		classes_num = 1
	else:
		# Multi-class classification
		classes_num = output.shape[1]  # Number of classes


	stats = []  # Compute predicted labels
	
	y_true = target.numpy()
	y_pred = torch.argmax(output, dim=1).numpy() if classes_num > 1 else (output > 0.5).numpy()

	# Calculate accuracy
	acc = metrics.accuracy_score(y_true, y_pred)
	
	# Compute confusion matrix
	conf_matrix = metrics.confusion_matrix(y_true, y_pred)
	
	
	# Iterate over classes or treat as binary if single output
	for k in range(classes_num):
		#print(classes_num)
		if classes_num == 1:
			# Binary classification
			recall = metrics.recall_score(y_true, y_pred)
			f1 = metrics.f1_score(y_true, y_pred)
			auc = metrics.roc_auc_score(y_true, output.numpy())
            # Specificity: TN / (TN + FP)
			if conf_matrix.shape == (2, 2):
				tn, fp, fn, tp = conf_matrix.ravel()
				specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
			else:
				specificity = None

			dict_stats = {
				'accuracy': acc,
				'recall': recall,
				'specificity': specificity,
				'f1': f1,
				'auc': auc,
				'confusion_matrix': conf_matrix.tolist()
            }
			stats.append(dict_stats)
		else:
			# Multi-class classification
			recall = metrics.recall_score((y_true == k), (y_pred == k))
			f1 = metrics.f1_score((y_true == k), (y_pred == k))
			auc = metrics.roc_auc_score((y_true == k), output[:, k].numpy())
			# Specificity for class k: TN / (TN + FP)
			cm = metrics.confusion_matrix((y_true == k), (y_pred == k))
			if cm.shape == (2, 2):
				tn, fp, fn, tp = cm.ravel()
				specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
			else:
				specificity = None

			dict_stats = {
				'accuracy': acc,
				'recall': recall,
				'specificity': specificity,
				'f1': f1,
				'auc': auc,
				'confusion_matrix': conf_matrix.tolist()
			}
			stats.append(dict_stats)


	return stats


# https://scikit-learn.org/stable/glossary.html#term-multi-class multiclass > 2 classes 
def calculate_patient_wise_metrics(audio_results_df, num_classes=None):
	"""
	Calculate patient-wise metrics based on audio predictions.

	Args:
		audio_results_df (pd.DataFrame): A DataFrame containing the following columns:
			- 'patient_id': ID of the patient
			- 'label': True label for the audio
			- 'pred_value': Predicted probabilities for the audio (binary or multiclass)
		num_classes (int, optional): The number of classes for multiclass classification.
									  If not provided, the function assumes binary classification.

	Returns:
		patient_results (pd.DataFrame): A DataFrame with patient-wise predictions and true labels.
		patient_metrics (dict): Dictionary containing overall patient-wise metrics (e.g., AUC, accuracy).
	"""

	# Group by patient_id and calculate mean predicted probabilities per patient
	patient_results = (
		audio_results_df.groupby('patient_id')
		.agg({
			'label': 'first',  # Assume all audios for a patient have the same true label
			'pred_value': lambda x: list(np.mean([p for p in x], axis=0)) # Mean of probabilities
		})
		.reset_index()
	)

	# For multiclass classification, pred_value will be a vector of probabilities.
	if num_classes is not None:  # Multiclass classification
		
		#  We take the argmax of the predicted probabilities to get the predicted class.
		patient_results['pred'] = patient_results['pred_value'].apply(lambda x: np.argmax(x))
	
	else:  # Binary classification (num_classes == 1)

		# Binary classification: threshold at 0.5 for predictions
		patient_results['pred'] = patient_results['pred_value'].apply(lambda x: 1 if x[1] > 0.5 else 0)

	# Calculate metrics
	y_true = patient_results['label']
	y_pred = patient_results['pred']
	y_pred_proba = np.array(patient_results['pred_value'].tolist())  # Ensure this is in the correct format for AUC
	
	# Compute confusion matrix
	conf_matrix = metrics.confusion_matrix(y_true, y_pred)
		# Calculate confusion matrix to get TN, FP, FN, TP for specificity
	
	tn, fp, fn, tp = conf_matrix.ravel()

	# Calculate specificity
	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
	patient_metrics = {
		'accuracy': metrics.accuracy_score(y_true, y_pred),
		'recall': metrics.recall_score(y_true, y_pred, average='macro' if num_classes is not None else 'binary'),
		'specificity': specificity,
		'f1_score': metrics.f1_score(y_true, y_pred, average='macro' if num_classes is not None else 'binary'),
		'auc': metrics.roc_auc_score(y_true, y_pred_proba[:, 1]),
		'precision': metrics.precision_score(y_true, y_pred, average='macro' if num_classes is not None else 'binary'),
		'confusion_matrix': conf_matrix.tolist()
	}

	return patient_results, patient_metrics


def save_experiment_results(
	exp_dir, 
	train_loss, 
	val_loss, 
	train_aucs, 
	val_aucs, 
	results_df, 
	patient_results, 
	patient_metrics, 
	audio_model, 
	optimizer, 
	args
):
	"""
	Saves training results, plots metrics, and stores all relevant experiment data.

	Parameters:
	exp_dir (str): Path to the experiment directory.
	train_loss (list): Training loss for each epoch.
	val_loss (list): Validation loss for each epoch.
	train_aucs (list): Training AUC for each epoch.
	val_aucs (list): Validation AUC for each epoch.
	results_df (pd.DataFrame): DataFrame with detailed results.
	patient_results (pd.DataFrame): DataFrame with patient-specific results.
	patient_metrics (dict): Dictionary of patient metrics.
	audio_model (torch.nn.Module): Trained model instance.
	optimizer (torch.optim.Optimizer): Optimizer used during training.
	args (dict): Experiment arguments.
	"""

	# Ensure the experiment directory exists
	os.makedirs(exp_dir, exist_ok=True)

	# Plot training and validation loss
	plt.figure(figsize=(10, 5))
	plt.plot(train_loss, label='Training Loss')
	plt.plot(val_loss, label='Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Training and Validation Loss')
	plt.legend()
	plt.savefig(f"{exp_dir}/loss.png", dpi = 200)
	plt.show()

	# Plot training and validation AUC
	plt.figure(figsize=(10, 5))
	plt.plot(val_aucs, label='Validation AUC', color='orange')
	plt.plot(train_aucs, label='Training AUC', color='blue')
	plt.xlabel('Epochs')
	plt.ylabel('AUC')
	plt.title('Training and Validation AUC')
	plt.legend()
	plt.savefig(f"{exp_dir}/AUC.png", dpi=200)
	plt.show()

	# Save results DataFrame to a CSV
	results_path = os.path.join(exp_dir, 'results_df.csv')
	results_df.to_csv(results_path, index=False, sep = "\t", decimal= ",")
	print(f"Results DataFrame saved to {results_path}")

	# Save patient results to a CSV
	patient_results_path = os.path.join(exp_dir, "patient_results.csv")
	patient_results.to_csv(patient_results_path, index=False, sep = "\t", decimal= ",")
	print(f"Patient results saved to {patient_results_path}")

	# Save patient metrics to a JSON file
	patient_metrics_path = os.path.join(exp_dir, "patient_metrics.json")
	with open(patient_metrics_path, "w") as f:
		json.dump(patient_metrics, f, indent=4)
	print(f"Patient metrics saved to {patient_metrics_path}")

	# Save training and validation metrics to a JSON file
	metrics_path = os.path.join(exp_dir, "loss_auc_metrics.json")
	metrics = {
	"val_loss": val_loss,
	"train_loss": train_loss,
	"val_aucs": val_aucs,
	"train_aucs": train_aucs
	}
	with open(metrics_path, "w") as f:
		json.dump(metrics, f, indent=4)
	print(f"Training and validation metrics saved to {metrics_path}")

	#if args['CV'] == False:	
		# Save model and optimizer states
	#models_dir = os.path.join(exp_dir, "models")
	#os.makedirs(models_dir, exist_ok=True)
		
	#model_path = os.path.join(models_dir, "audio_model.pth")
	#optim_path = os.path.join(models_dir, "optim_state.pth")

	#torch.save(audio_model.state_dict(), model_path)
	#torch.save(optimizer.state_dict(), optim_path)

	#print(f"Model saved to {model_path}")
	#print(f"Optimizer state saved to {optim_path}")

	# Save experiment arguments to a JSON file
	args_path = os.path.join(exp_dir, "args.json")
	with open(args_path, "w") as f:
		json.dump(args, f, indent=4)
	print(f"Experiment arguments saved to {args_path}")
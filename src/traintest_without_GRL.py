# -*- coding: utf-8 -*-
# @Time    : 21/1/25 10:48 AM
# @Author  : Marcos Aguilella
# @Affiliation  : IDIVAL
# @Email   : marcos.aguilella@idival.org
# @File    : traintest.py

from models import * 
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import shutil
from utilities import *
from dataloader.audio_dataset_class import *
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix 

def compute_class_weights(labels):
	class_counts = torch.bincount(labels)
	total_samples = labels.size(0)
	weights = total_samples / (len(class_counts) * class_counts)
	return weights


def use_tiebreaker(current_metrics, best_metrics, tiebreaker_metric):
	"""
	Break ties using secondary metric.
	
	Args:
		current_metrics (dict): Current epoch metrics
		best_metrics (dict): Best epoch metrics so far
		tiebreaker_metric (str): Name of tiebreaker metric
	
	Returns:
		bool: True if current is better than best based on tiebreaker
	"""
	current_tb = current_metrics.get(tiebreaker_metric, -float('inf'))
	best_tb = best_metrics.get(tiebreaker_metric, -float('inf'))
	
	# Tiebreaker is typically val_loss (minimize) or accuracy (maximize)
	minimize_tb = (tiebreaker_metric == 'val_loss')
	
	if minimize_tb:
		return current_tb < best_tb - 1e-6
	else:
		return current_tb > best_tb + 1e-6


def is_better_model(current_metrics, best_metrics, args):
	"""
	Compare current epoch metrics with best metrics.
	Returns True if current is better than best.
	
	Logic:
	1. Compare PRIMARY metric first
	2. If equal (within tolerance), use TIEBREAKER metric
	
	Args:
		current_metrics (dict): Current epoch metrics {'metric_name': value, ...}
		best_metrics (dict): Best epoch metrics {'metric_name': value, ...}
		args (dict): Contains 'best_model_metric' and 'best_model_tiebreaker'
	
	Returns:
		bool: True if current is better than best
	"""
	primary_metric = args['best_model_metric']
	tiebreaker_metric = args['best_model_tiebreaker']
	
	# Get metric values
	current_primary = current_metrics.get(primary_metric, -float('inf'))
	best_primary = best_metrics.get(primary_metric, -float('inf'))
	
	# Determine if metric should be minimized (val_loss) or maximized (accuracy, AUC)
	minimize_primary = (primary_metric == 'val_loss')
	
	# Compare primary metrics
	if minimize_primary:
		# Lower is better (loss)
		if current_primary < best_primary - 1e-6:
			return True
		elif abs(current_primary - best_primary) < 1e-6:
			# Primary metrics are equal → use tiebreaker
			return use_tiebreaker(current_metrics, best_metrics, tiebreaker_metric)
		else:
			return False
	else:
		# Higher is better (accuracy, AUC)
		if current_primary > best_primary + 1e-6:
			return True
		elif abs(current_primary - best_primary) < 1e-6:
			# Primary metrics are equal → use tiebreaker
			return use_tiebreaker(current_metrics, best_metrics, tiebreaker_metric)
		else:
			return False


def save_best_model_checkpoint(audio_model, epoch_metrics, exp_dir, epoch):
	"""
	Save best model checkpoint (weights only, no optimizer state).
	
	Args:
		audio_model: PyTorch model
		epoch_metrics (dict): Metrics from this epoch
		exp_dir (str): Directory to save checkpoint
		epoch (int): Current epoch number
	"""
	# Create checkpoint directory if it doesn't exist
	checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
	os.makedirs(checkpoint_dir, exist_ok=True)
	
	# Handle DataParallel wrapper
	if isinstance(audio_model, nn.DataParallel):
		model_state = audio_model.module.state_dict()
	else:
		model_state = audio_model.state_dict()
	
	# Create checkpoint (weights + metrics only, no optimizer)
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model_state,
		'metrics': epoch_metrics
	}
	
	# Save checkpoint
	checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
	torch.save(checkpoint, checkpoint_path)
	
	print(f"✓ Saved best model at epoch {epoch+1} | "
		  f"{list(epoch_metrics.keys())[0]}={list(epoch_metrics.values())[0]:.4f}")


def load_best_model_and_evaluate(audio_model, exp_dir, val_loader, args, device):
	"""
	Load best model checkpoint and evaluate on validation set (AC patients).
	
	Args:
		audio_model: PyTorch model
		exp_dir (str): Experiment directory
		val_loader: Validation data loader
		args (dict): Configuration
		device: torch.device
	
	Returns:
		tuple: (stats, val_loss, results_df, patient_results, patient_metrics, best_epoch)
	"""
	checkpoint_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
	
	if not os.path.exists(checkpoint_path):
		print("⚠ No best model checkpoint found, using current model")
		# Evaluate current model
		stats, val_loss, results_df, patient_results, patient_metrics = \
			validate_without_GRL(audio_model, val_loader, args)
		return stats, val_loss, results_df, patient_results, patient_metrics, -1
	
	# Load checkpoint
	checkpoint = torch.load(checkpoint_path, map_location=device)
	best_epoch = checkpoint['epoch']
	best_metrics = checkpoint['metrics']
	
	# Load model weights
	if isinstance(audio_model, nn.DataParallel):
		audio_model.module.load_state_dict(checkpoint['model_state_dict'])
	else:
		audio_model.load_state_dict(checkpoint['model_state_dict'])
	
	print(f"\n✓ Loaded best model from epoch {best_epoch+1}")
	print(f"  Best metrics: {best_metrics}")
	
	# Evaluate loaded model on validation set (AC patients)
	audio_model.eval()
	stats, val_loss, results_df, patient_results, patient_metrics = \
		validate_without_GRL(audio_model, val_loader, args)
	
	print(f"✓ Evaluated best model on validation set (AC patients)")
	print(f"  Validation Loss: {val_loss:.4f}")
	print(f"  Patient-wise Accuracy: {patient_metrics['accuracy']:.4f}")
	print(f"  Patient-wise AUC: {patient_metrics['auc']:.4f}")
	
	return stats, val_loss, results_df, patient_results, patient_metrics, best_epoch


def initialize_model(args):
	"""
	Initialize the audio model based on the provided arguments.
	
	Args:
		args (dict): Configuration dictionary containing model parameters.
	
	Returns:
		audio_model: Initialized PyTorch model.
	"""
	if args['model_type'] == 'EffNetAttention':
		audio_model = EffNetAttention(label_dim=2, b=args['b'], pretrain=True, head_num=args['head_num'], dropout_rate=args['dropout_rate'], dropatt_rate = args['dropatt_rate'], use_efficientnetv2=args['use_efficientnetv2'], v2_model_name = args['v2_model_name'])
	elif args['model_type'] == 'ResNetAttention':
		audio_model = ResNetAttention(label_dim=2, pretrain=args['pretrain'], dropatt_rate = args['dropatt_rate'], dropout_rate=args['dropout_rate'])
	# elif args['model_type'] == 'ASTModel':
	# 	audio_model = ASTModel(label_dim=2,fstride=10, tstride=10, input_fdim=128, input_tdim=313, imagenet_pretrain=args['pretrain'], audioset_pretrain=args['pretrain'],  dropout_type="attention", mlp_dropout_p=args['dropout_rate'], attention_dropout_p=args['dropout_rate'], pretrained_models_path="/home/marcos/Documentos/GitHub/TFM_code/pretrained_models")
	
	return audio_model

# Training function with validation
def train_without_GRL(audio_model, train_loader, val_loader, args):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('\nRunning on ' + str(device))
	
	# Create or clear the experiment directory
	exp_dir = args['exp_dir']
	CV=args['CV']
	if os.path.exists(exp_dir) and CV==False :
		# Clear the directory
		print("Removing existing files from directory: ", exp_dir)
		shutil.rmtree(exp_dir)
        
	elif not os.path.exists(exp_dir) and CV == False:
		# Create the directory
		print("Creating directory: ", exp_dir)
		os.makedirs(exp_dir)

	torch.set_grad_enabled(True)

	# Trackers for statistics
	batch_time = AverageMeter()
	data_time = AverageMeter()
	loss_meter = AverageMeter()
	global_step, epoch = 0, 0
	start_time = time.time()

	# Optimizer and loss function setup
	optimizer = optim.Adam(audio_model.parameters(), lr=args['lr'], weight_decay=1e-4, betas=(0.95, 0.999))

	# Collect all labels from the training set to compute class weights
	all_train_labels = []
	for _, labels, _ in train_loader:
		all_train_labels.append(labels)
	all_train_labels = torch.cat(all_train_labels)
	class_weights = compute_class_weights(all_train_labels)
	class_weights = class_weights.to(device)

	#loss_fn = nn.BCEWithLogitsLoss() if args['loss'] == 'BCE' else nn.CrossEntropyLoss()

	if args['loss'] == 'BCE':
		loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1], label_smoothing = 0.1)  # For binary classification
	else:
		loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing = 0.1)  # For multi-class classification

	scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
											   list(range(args['lrscheduler_start'],
											   args['n_epochs'], args['lrscheduler_step'])),
											   gamma=args['lrscheduler_decay'])

	# For mixed precision training
	scaler = torch.amp.GradScaler()

	print("Starting training...")
	print(f"Experiment directory: {exp_dir}")
	audio_model.train()

	# Lists to store metrics for plotting
	train_loss = []
	train_aucs = []
	train_pm_aucs = []
	val_loss = []
	val_aucs = []
	val_pm_aucs = []
	
	# Early stopping initialization
	best_metric = -float('inf') if args['monitor_metric'] != 'val_loss' else float('inf')
	best_epoch_early_stop = 0
	early_stop_counter = 0
	
	# Best model tracking initialization
	best_model_metrics = None
	best_model_epoch = 0
	skip_epochs = args.get('skip_epochs_for_best', 5)  # Default to 5 if not specified

	# Start training loop
	while epoch < args['n_epochs']:

		begin_time = time.time()
		audio_model.train()
		loss_meter.reset()

		all_outputs = []
		all_targets = []
		results = []


		for i, (audio_input, labels, indices) in enumerate(train_loader):
			B = audio_input.size(0)
			audio_input = audio_input.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)


			# Timing data loading
			data_time.update(time.time() - begin_time)

			with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
				raw_audio_output = audio_model(audio_input)

				# Compute the loss based on the model's output and true labels
				if isinstance(loss_fn, nn.CrossEntropyLoss):
					PD_loss = loss_fn(raw_audio_output, labels.long())
					audio_output = torch.softmax(raw_audio_output, dim=1)
				else:
					if len(labels.shape) == 1:
						labels = labels.unsqueeze(1)
					PD_loss = loss_fn(raw_audio_output, labels.float())
					audio_output = torch.sigmoid(raw_audio_output)

			optimizer.zero_grad()
			scaler.scale(PD_loss).backward()
			scaler.step(optimizer)
			scaler.update()

			loss_meter.update(PD_loss.item(), B)

			# Store outputs and targets for AUC calculation
			all_outputs.append(audio_output.detach().cpu())
			all_targets.append(labels.detach().cpu())

			for idx, patient_id in enumerate(indices):
				true_label = labels[idx].item()

				if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
					# For CrossEntropyLoss: predicted class is the argmax
					pred_value = audio_output[idx].tolist()
					pred_class = torch.argmax(audio_output[idx]).item()   
				else:
					# For BCEWithLogitsLoss or similar: binary case
					pred_value = audio_output[idx].item()
					pred_class = 1 if pred_value > 0.5 else 0

				# Save patient_id, true_label, pred_class, and pred_value
				results.append({
					'patient_id': patient_id,
					'label': true_label,
					'pred_value': pred_value,  # Raw prediction value
					'pred': pred_class		# Final predicted class (binary or multi-class)
					})


			batch_time.update(time.time() - begin_time)
			global_step += 1

		# Learning rate scheduling
		scheduler.step()

		# Calculate AUC and accuracy for the training set
		all_outputs = torch.cat(all_outputs)
		all_targets = torch.cat(all_targets)

		train_stats = calculate_stats(all_outputs, all_targets)
		train_auc = train_stats[1]['auc'] if train_stats else 0
		train_acc = train_stats[0]['accuracy'] if train_stats else 0
		
		results_df = pd.DataFrame(results)
		patient_results, patient_metrics = calculate_patient_wise_metrics(results_df)
		patient_metric_AUC = patient_metrics['auc']
		patient_metric_acc = patient_metrics['accuracy']
		
		# Store training loss for this epoch
		train_loss.append(loss_meter.avg)
		train_aucs.append(train_auc)
		train_pm_aucs.append(patient_metric_AUC)
		
		# Perform validation and store metrics
		stats, val_loss_epoch, results_df, patient_results, patient_metrics = validate_without_GRL(audio_model, val_loader, args)
		test_AUC = stats[1]['auc'] if stats else 0
		test_acc = stats[0]['accuracy'] if stats else 0
		test_pm_AUC = patient_metrics['auc']
		test_pm_acc = patient_metrics['accuracy']
		val_loss.append(val_loss_epoch)
		val_aucs.append(test_AUC)
		val_pm_aucs.append(test_pm_AUC)
		
		# Determine current metric value
		if args['monitor_metric'] == 'val_acc':
			current_metric = stats[0]['accuracy'] if stats else 0
		elif args['monitor_metric'] == 'val_pm_acc':
			current_metric = patient_metrics['accuracy']
		else:  # val_loss
			current_metric = val_loss_epoch

		# Print training and validation metrics
		print(f"Epoch {epoch + 1}/{args['n_epochs']} - Training Loss: {loss_meter.avg:.4f} - Training acc: {train_acc:.3f}, Training PW acc: {patient_metric_acc:.3f}, "
			  f"Val Loss: {val_loss_epoch:.4f}, Val acc: {test_acc:.3f}, Test PW acc: {test_pm_acc:.3f}, Training time: {time.time() - begin_time:.1f}s")

		# Build epoch metrics dictionary for best model tracking
		epoch_metrics = {
			'test_pm_acc': test_pm_acc,
			'val_loss': val_loss_epoch,
			'test_pm_auc': test_pm_AUC,
			'test_acc': test_acc,
			'train_loss': loss_meter.avg,
		}
		
		# Track best model (skip first N epochs)
		if args.get('save_best_model', False):
			if epoch >= skip_epochs:
				if best_model_metrics is None or is_better_model(epoch_metrics, best_model_metrics, args):
					best_model_metrics = epoch_metrics.copy()
					best_model_epoch = epoch
					save_best_model_checkpoint(audio_model, epoch_metrics, exp_dir, epoch)

		# Check for improvement
		is_better = False
		if args['monitor_metric'] == 'val_loss':
			if current_metric < (best_metric - args['min_delta']):
				is_better = True
		else:  # For AUC metrics (higher is better)
			if current_metric > (best_metric + args['min_delta']):
				is_better = True

		# Update best metric and check early stopping
		if is_better:
			best_metric = current_metric
			best_epoch_early_stop = epoch
			early_stop_counter = 0
		else:
			early_stop_counter += 1
			if early_stop_counter >= args['early_stop_patience']:
				print(f"\nEarly stopping triggered at epoch {epoch+1}!")
				print(f"No improvement for {args['early_stop_patience']} epochs")
				print(f"Best {args['monitor_metric']}: {best_metric:.4f} at epoch {best_epoch_early_stop+1}")
				break

		epoch += 1

	# Load best model and evaluate on AC patients (if best model tracking is enabled)
	if args.get('save_best_model', False):
		stats, val_loss_final, results_df, patient_results, patient_metrics, best_model_epoch = \
			load_best_model_and_evaluate(audio_model, exp_dir, val_loader, args, device)
		# val_loss_value = val_loss_final  # Update val_loss with best model evaluation
	else:
		best_model_epoch = -1

	if CV==False:
		save_experiment_results(exp_dir=args['exp_dir'], train_loss=train_loss, val_loss=val_loss, train_aucs=train_aucs, val_aucs=val_aucs, results_df=results_df, patient_results=patient_results, patient_metrics=patient_metrics, audio_model=audio_model, optimizer=optimizer, args=args, best_epoch=best_model_epoch if args.get('save_best_model', False) else -1
	)
	
	final_time = time.time() - start_time
	print(f"\nTraining completed in {final_time/60:.2f} minutes")
	
	return stats, val_loss, val_aucs, results_df, patient_results, patient_metrics, train_aucs, train_loss, best_model_epoch


# Validation function
def validate_without_GRL(audio_model, val_loader, args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	batch_time = AverageMeter()
	loss_meter_val = AverageMeter()

	if isinstance(audio_model, nn.DataParallel):
		audio_model = audio_model.module
	audio_model = audio_model.to(device)

	# switch to evaluate mode
	audio_model.eval()

	A_outputs, A_targets = [], []  # Lists to store outputs and targets

	results = []  # To store detailed metrics for each sample

	# Set loss function
	if args['loss'] == 'BCE':
		loss_fn = nn.BCEWithLogitsLoss()
	elif args['loss'] == 'CE':
		loss_fn = nn.CrossEntropyLoss()
	else:
		raise ValueError("Unsupported loss function")

	with torch.no_grad():
		for audio_input, labels, indices in val_loader:
			audio_input, labels = audio_input.to(device, non_blocking=True), labels.to(device, non_blocking=True)


			# compute output
			raw_audio_output = audio_model(audio_input)

			# compute the loss
			if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
				PD_loss = loss_fn(raw_audio_output, labels.long())
				audio_output = torch.softmax(raw_audio_output, dim=1)
			else:
				if len(labels.shape) == 1:
					labels = labels.unsqueeze(1)
				PD_loss = loss_fn(raw_audio_output, labels.float())
				audio_output = torch.sigmoid(raw_audio_output)


			loss_meter_val.update(PD_loss.item(), labels.size(0))  # Update loss meter

			A_outputs.append(audio_output)
			A_targets.append(labels)

			# Iterate through each sample in the batch and store results
			for idx, patient_id in enumerate(indices):
				true_label = labels[idx].item()

				if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
					# For CrossEntropyLoss: predicted class is the argmax
					pred_value = audio_output[idx].tolist()
					pred_class = torch.argmax(audio_output[idx]).item()   
				else:
					# For BCEWithLogitsLoss or similar: binary case
					pred_value = audio_output[idx].item()
					pred_class = 1 if pred_value > 0.5 else 0

				# Save patient_id, true_label, pred_class, and pred_value
				results.append({
					'patient_id': patient_id,
					'label': true_label,
					'pred_value': pred_value, # Raw prediction value
					'pred': pred_class # Final predicted class (binary or multi-class)
					})
		targets = torch.cat(A_targets)
		outputs = torch.cat(A_outputs)
		# Calculate statistics
		stats = calculate_stats(outputs, targets)

		results_df = pd.DataFrame(results)
		patient_results, patient_metrics = calculate_patient_wise_metrics(results_df)

	return stats, loss_meter_val.avg, results_df, patient_results, patient_metrics


def cross_validate_and_save(
    audio_segments, labels_np, patient_ids, args, audio_segments_AC, labels_AC, patient_ids_AC, 
):
    """
    Perform cross-validation, train the model for each fold, and save results.

    Parameters:
        audio_segments (ndarray): Audio features (X).
        labels_np (ndarray): Labels (y).
        patient_ids (ndarray): Patient IDs for group-based splitting.
        args (dict): Experiment arguments.
        audio_model (nn.Module): The audio model to train and validate.
        audio_segments_AC (ndarray): Audio features for AC dataset.
        labels_AC (ndarray): Labels for AC dataset.
        patient_ids_AC (ndarray): Patient IDs for AC dataset.
    """

    # # Step 1: Create a mapping of patient IDs to their diagnosis (PD = 1, HC = 0)
    # patient_label_map = {}
    # for patient_id, label in zip(patient_ids, labels_np):
    #     if patient_id not in patient_label_map:
    #         patient_label_map[patient_id] = label  # Assign the first encountered label (assume consistent labels)

    # # Step 2: Extract unique patients and their corresponding labels
    # unique_patients = np.array(list(patient_label_map.keys()))  # Unique patient IDs
    # patient_labels = np.array([patient_label_map[pid] for pid in unique_patients])  # PD = 1, HC = 0

	# Full arrays
    X = np.array(audio_segments)
    y = np.array(labels_np)
    groups = np.array(patient_ids)

    overall_results = pd.DataFrame()
    patient_wise_results = pd.DataFrame()
    patient_wise_metrics = pd.DataFrame()
    patient_wise_AC_metrics = pd.DataFrame()
    fold_metrics = []
    
    # Track AC patient probabilities across folds: {patient_id: [prob_fold1, prob_fold2, ...]}
    ac_patient_probs_across_folds = {}
    
    n_splits = args['n_splits']  # Number of CV folds
    # splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=args['random_state_split'])
    # splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=40)
    splitter = StratifiedGroupKFold(n_splits=args['n_splits'], shuffle=True, random_state=args['random_state_split'])

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        torch.cuda.empty_cache()
        print("---------------")
        print(f"Starting Fold {fold_idx + 1}")
        print("---------------")

        # Set up experiment directory for the current fold
        fold_exp_dir = os.path.join(args['exp_dir'], f'fold_{fold_idx + 1}')
        if os.path.exists(fold_exp_dir):
            print(f"Clearing existing directory: {fold_exp_dir}")
            shutil.rmtree(fold_exp_dir)
        os.makedirs(fold_exp_dir)

        train_patients = np.unique(groups[train_idx])
        test_patients = np.unique(groups[test_idx])

        # Count PD & HC in the training set
        train_PD_count = np.sum(y[train_idx] == 1)
        train_HC_count = np.sum(y[train_idx] == 0)
        test_HC_count = np.sum(y[test_idx] == 0)
        test_PD_count = np.sum(y[test_idx] == 1)

        print(f"Fold {fold_idx + 1}: Final Train PD={train_PD_count}, Train HC={train_HC_count}")
        print(f"Fold {fold_idx + 1}: Test PD={test_PD_count}, Test HC={test_HC_count}")

        # Step 5: Assign audio segments based on selected patients
        # train_indices = np.array([i for i, pid in enumerate(patient_ids) if pid in train_patients])
        # test_indices = np.array([i for i, pid in enumerate(patient_ids) if pid in test_patients])

        # Split the arrays using the indices
        X_train = X[train_idx]
        y_train = y[train_idx]
        patient_ids_train = groups[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        test_patient_ids = groups[test_idx]

        if args['oversampling'] == True:
            X_train_oversampled, y_train_oversampled, train_patient_ids_oversampled, _  =  utils_audio_PD_project.oversample_training_data(X_train, y_train, patient_ids_train, domain_labels=None, GRL = False)

            print(f"Train labels distribution (before oversampling): {np.bincount(y_train)}")
            print(f"Train labels distribution (after oversampling): {np.bincount(y_train_oversampled)}")
            print(f"Test labels distribution: {np.bincount(y_test)}")

        else:
            X_train_oversampled, y_train_oversampled, train_patient_ids_oversampled = X_train, y_train, patient_ids_train
            print(f"Train labels distribution: {np.bincount(y_train)}")


        # First create the dataset without normalization and calculate the metrics
        train_dataset = ParkinsonAudioDataset_without_GRL(X_train_oversampled, y_train_oversampled, train_patient_ids_oversampled, n_mels=args['n_mels'], sr=args['sr'], hop_length = args['hop_length'], n_fft = args['n_fft'], win_length = args['win_length'],  normalize=False, augment = True, freq_mask = args['freq_mask'], time_mask = args['time_mask'], std=args['std'], noise = args['noise'], mixup_rate = args['mixup_rate'], augmentation_rate= args['augmentation_rate'])
        # train_dataset = dataloader.ParkinsonAudioDataset_without_GRL(X_train, y_train, patient_ids_train, n_mels=args['n_mels'], sr=args['sr'], hop_length = args['hop_length'], n_fft = args['n_fft'], win_length = args['win_length'],  normalize=False, augment = True, freq_mask = args['freq_mask'], time_mask = args['time_mask'], std=args['std'], noise = args['noise'], mixup_rate = args['mixup_rate'], augmentation_rate= args['augmentation_rate'])
        dataset_mean, dataset_std = calculate_mean_std(train_dataset, GRL = False)
        
		# print(dataset_mean, dataset_std)
        args['dataset_mean'] = dataset_mean
        args['dataset_std'] = dataset_std

        # Now create the dataset with normalization
        # train_dataset = dataloader.ParkinsonAudioDataset_without_GRL(X_train, y_train, patient_ids_train, n_mels=args['n_mels'], sr=args['sr'], hop_length = args['hop_length'], n_fft = args['n_fft'], win_length = args['win_length'], augment = True, normalize=True, transpose = True, norm_mean=dataset_mean, norm_std=dataset_std, freq_mask = args['freq_mask'], time_mask = args['time_mask'], std = args['std'], noise = args['noise'], mixup_rate = args['mixup_rate'], augmentation_rate = args['augmentation_rate'])
        train_dataset = ParkinsonAudioDataset_without_GRL(X_train_oversampled, y_train_oversampled, train_patient_ids_oversampled, n_mels=args['n_mels'], sr=args['sr'], hop_length = args['hop_length'], n_fft = args['n_fft'], win_length = args['win_length'], augment = True, normalize=True, transpose = True, norm_mean=dataset_mean, norm_std=dataset_std, freq_mask = args['freq_mask'], time_mask = args['time_mask'], std = args['std'], noise = args['noise'], mixup_rate = args['mixup_rate'], augmentation_rate = args['augmentation_rate'])
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True)

        # Create DataLoader for testing
        test_dataset = ParkinsonAudioDataset_without_GRL(X_test, y_test, test_patient_ids, n_mels=args['n_mels'], sr=args['sr'], hop_length = args['hop_length'], n_fft = args['n_fft'], win_length = args['win_length'], augment = False, normalize=True, transpose = True, norm_mean=dataset_mean, norm_std=dataset_std, std=args['std'])
        test_loader = DataLoader(test_dataset, args['batch_size'], shuffle=False, pin_memory=True)


        # Initialize a fresh model
        torch.cuda.empty_cache()
        audio_model_fold = initialize_model(args)
        audio_model_fold = audio_model_fold.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Save original exp_dir
        # original_exp_dir = args['exp_dir']
        # Update args with fold-specific directory
        # args['exp_dir'] = fold_exp_dir

        # Train the model
        stats, val_loss, val_aucs, results_df, patient_results, patient_metrics, train_aucs, train_loss, best_model_epoch = train_without_GRL(audio_model_fold, train_loader, test_loader, args)

        # Save results for the current fold
        save_experiment_results(
            exp_dir=fold_exp_dir,
            train_loss=train_loss,
            val_loss=val_loss,
            train_aucs=train_aucs,
            val_aucs=val_aucs,
            results_df=results_df,
            patient_results=patient_results,
            patient_metrics=patient_metrics,
            audio_model=audio_model_fold,
            optimizer=None,  # If optimizer is used, pass it here
            args=args
        )

        # Restore original exp_dir for next fold
        # args['exp_dir'] = original_exp_dir

        # Append fold results to overall DataFrames
        patient_results['fold'] = fold_idx + 1
        results_df['fold'] = fold_idx + 1
        #patient_metrics_df = pd.DataFrame(patient_metrics, index=[0])
        patient_metrics_df = pd.DataFrame([patient_metrics])
        patient_metrics_df['fold'] = fold_idx + 1

        overall_results = pd.concat([overall_results, results_df], ignore_index=True)
        patient_wise_results = pd.concat([patient_wise_results, patient_results], ignore_index=True)
        patient_wise_metrics = pd.concat([patient_wise_metrics, patient_metrics_df], ignore_index=True)

        # Extract metrics for the current fold
        acc = metrics.accuracy_score(patient_results['label'], patient_results['pred'])
        auc = metrics.roc_auc_score(patient_results['label'], [x[1] for x in patient_results['pred_value']])
        precision =  metrics.precision_score(patient_results['label'], patient_results['pred'], average='macro')
        recall = metrics.recall_score(patient_results['label'], patient_results['pred'], average='macro')

        # Calculate confusion matrix to get TN, FP, FN, TP for specificity
        cm = confusion_matrix(patient_results['label'], patient_results['pred'])
        tn, fp, fn, tp = cm.ravel()

        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        fold_metrics.append({'fold': fold_idx + 1, 'Accuracy': acc, 'AUC': auc, 'Precision': precision, 'Recall': recall, 'Specificity': specificity})

        print(f"Fold {fold_idx + 1} Metrics - Accuracy: {acc:.4f}, AUC: {auc:.4f}, Specificity: {specificity:.4f}")

        # --- AC EVALUATION (after test set evaluation in each fold) ---

        # Create DataLoader for AC patients (use the same normalization as training)
        val_dataset_AC = ParkinsonAudioDataset_without_GRL(
            audio_segments_AC, labels_AC, patient_ids_AC,
            n_mels=args['n_mels'], sr=args['sr'], hop_length=args['hop_length'],
            n_fft=args['n_fft'], win_length=args['win_length'],
            augment=False, normalize=True, transpose=True,
            norm_mean=dataset_mean, norm_std=dataset_std, std=args['std']
        )
        val_loader_AC = DataLoader(val_dataset_AC, batch_size=16, shuffle=False, pin_memory=True)

        # Evaluate on AC patients
        stats_AC, val_loss_epoch_AC, results_df_AC, patient_results_AC, patient_metrics_AC = validate_without_GRL(
            audio_model_fold, val_loader_AC, args
    )

        # Save per audio AC results for this fold
        ac_results_path = os.path.join(fold_exp_dir, 'AC_results.csv')
        results_df_AC.to_csv(ac_results_path, index=False, sep = "\t", decimal= ",")

        # Save patient-wise AC results for this fold
        ac_pw_results_path = os.path.join(fold_exp_dir, 'AC_PW_results.csv')
        patient_results_AC.to_csv(ac_pw_results_path, index=False, sep = "\t", decimal= ",")

        ac_patient_metrics_path = os.path.join(fold_exp_dir, 'AC_patient_metrics.csv')
        patient_metrics_AC_df = pd.DataFrame([patient_metrics_AC])
        patient_metrics_AC_df.to_csv(ac_patient_metrics_path, index=False, sep = "\t", decimal= ",")
        patient_metrics_AC_df['fold'] = fold_idx + 1


        patient_metrics_AC_df = pd.DataFrame([patient_metrics_AC])
        patient_wise_AC_metrics = pd.concat([patient_wise_AC_metrics, patient_metrics_AC_df], ignore_index=True)

        # Extract AC patient probabilities for this fold (probability of class 1 - PD class)
        for idx, row in patient_results_AC.iterrows():
            patient_id = row['patient_id']
            # Extract probability of class 1 (PD class)
            prob_class_1 = row['pred_value'][1] if isinstance(row['pred_value'], (list, tuple)) else row['pred_value']
            
            if patient_id not in ac_patient_probs_across_folds:
                ac_patient_probs_across_folds[patient_id] = []
            ac_patient_probs_across_folds[patient_id].append(prob_class_1)

        # print(f"AC evaluation results saved to {ac_results_path} and {ac_patient_metrics_path}")



    # Save overall results across folds
    overall_results_path = os.path.join(args['exp_dir'], 'overall_results.csv')
    overall_results.to_csv(overall_results_path, index=False, sep = "\t", decimal= ",")
    # print(f"Overall results saved to {overall_results_path}")

    fold_metrics_path = os.path.join(args['exp_dir'], 'cv_metrics.csv')
    fold_metrics_df = pd.DataFrame(fold_metrics)

    fold_AC_metrics_path = os.path.join(args['exp_dir'], 'cv_AC_metrics.csv')
    fold_AC_metrics_df = pd.DataFrame(patient_wise_AC_metrics)
    
    # Calculate the mean of the metrics
    mean_metrics_AC = fold_AC_metrics_df.mean(numeric_only=True).to_dict()
    #mean_metrics_AC['fold'] = 'Mean' # Add a label for the mean row

    fold_AC_metrics_df = pd.concat([fold_AC_metrics_df, pd.DataFrame([mean_metrics_AC])], ignore_index=True)
    
    # Add AC patient probability columns (alphabetically sorted)
    if ac_patient_probs_across_folds:
        # Sort patient IDs alphabetically
        sorted_patient_ids = sorted(ac_patient_probs_across_folds.keys())
        
        for patient_id in sorted_patient_ids:
            probs = ac_patient_probs_across_folds[patient_id]
            # Add probabilities for each fold row, and mean probability for the last (mean) row
            for fold_idx in range(len(probs)):
                fold_AC_metrics_df.at[fold_idx, patient_id] = probs[fold_idx]
            
            # Add mean probability for the last row (mean row)
            mean_prob = np.mean(probs)
            fold_AC_metrics_df.at[len(probs), patient_id] = mean_prob
    
    fold_AC_metrics_df.to_csv(fold_AC_metrics_path, index=False, sep = "\t", decimal= ",")

    # Calculate the mean of the metrics
    mean_metrics = fold_metrics_df.mean(numeric_only=True).to_dict()
    mean_metrics['fold'] = 'Mean' # Add a label for the mean row

    # Append the mean row to the DataFrame
    fold_metrics_df = pd.concat([fold_metrics_df, pd.DataFrame([mean_metrics])], ignore_index=True)

    fold_metrics_df.to_csv(fold_metrics_path, index=False, sep = "\t", decimal= ",")
    # print(f"Cross-validation metrics saved to {fold_metrics_path}")
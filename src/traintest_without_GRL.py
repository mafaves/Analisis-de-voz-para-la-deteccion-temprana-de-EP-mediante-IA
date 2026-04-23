# -*- coding: utf-8 -*-
# @Time    : 21/1/25 10:48 AM
# @Author  : Marcos Aguilella
# @Affiliation  : IDIVAL
# @Email   : marcos.aguilella@idival.org
# @File    : traintest.py

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import shutil
from utilities import *

def compute_class_weights(labels):
	class_counts = torch.bincount(labels)
	total_samples = labels.size(0)
	weights = total_samples / (len(class_counts) * class_counts)
	return weights

  
# Training function with validation
def train_without_GRL(audio_model, train_loader, val_loader, args):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Running on ' + str(device))
	# Create or clear the experiment directory
	exp_dir = args['exp_dir']
	CV=args['CV']
	if os.path.exists(exp_dir) and CV==False :
		# Clear the directory
		print("Removing existing files from directory ", exp_dir)
		shutil.rmtree(exp_dir)
        
	elif not os.path.exists(exp_dir) and CV == False:
		# Create the directory
		print("Creating directory ", exp_dir)
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

	print("Starting training without GRL...")
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
	best_epoch = 0
	early_stop_counter = 0

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
			  f"Val Loss: {val_loss_epoch:.4f}, Val acc: {test_acc:.3f}, Test PW acc: {test_pm_acc:.3f}, Training time: {time.time() - begin_time:.3f}s")


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
			best_epoch = epoch
			early_stop_counter = 0
		else:
			early_stop_counter += 1
			if early_stop_counter >= args['early_stop_patience']:
				print(f"\nEarly stopping triggered at epoch {epoch+1}!")
				print(f"No improvement for {args['early_stop_patience']} epochs")
				print(f"Best {args['monitor_metric']}: {best_metric:.4f} at epoch {best_epoch+1}")
				break

		epoch += 1

	if CV==False:
		save_experiment_results(exp_dir=args['exp_dir'], train_loss=train_loss, val_loss=val_loss, train_aucs=train_aucs, val_aucs=val_aucs, results_df=results_df, patient_results=patient_results, patient_metrics=patient_metrics, audio_model=audio_model, optimizer=optimizer, args=args
	)
	return stats, val_loss, val_aucs, results_df, patient_results, patient_metrics, train_aucs, train_loss


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
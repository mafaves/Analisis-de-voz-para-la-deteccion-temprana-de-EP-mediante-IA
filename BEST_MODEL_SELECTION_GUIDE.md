# Best Model Selection Implementation Guide

## Overview

The `traintest_without_GRL.py` module has been enhanced with automatic best model selection functionality. This allows you to:

1. **Track the best model** based on a primary metric (e.g., `test_pm_acc`)
2. **Break ties** using a secondary metric (e.g., `val_loss`)
3. **Skip early epochs** to avoid random fluctuations
4. **Automatically evaluate AC patients** with the best model
5. **Save only model weights** (lightweight checkpoints)

## New Parameters

Add these parameters to your `args` dictionary:

```python
args['save_best_model'] = True                    # Enable feature
args['best_model_metric'] = 'test_pm_acc'         # Primary metric (higher is better)
args['best_model_tiebreaker'] = 'val_loss'        # Tiebreaker metric (lower is better)
args['skip_epochs_for_best'] = 5                  # Skip first N epochs
```

### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_best_model` | bool | False | Enable/disable best model tracking |
| `best_model_metric` | str | None | Primary metric to optimize (e.g., 'test_pm_acc', 'val_loss', 'test_pm_auc') |
| `best_model_tiebreaker` | str | None | Secondary metric for tie-breaking (e.g., 'val_loss', 'test_acc') |
| `skip_epochs_for_best` | int | 5 | Number of initial epochs to skip |

## How It Works

### Metric Comparison Logic

1. **Primary Metric Comparison**: The best model is selected based on the primary metric
   - If primary metric is `val_loss` → **lower is better**
   - If primary metric is any accuracy/AUC → **higher is better**

2. **Tie-Breaking**: If two epochs have identical primary metric (within tolerance of 1e-6)
   - Uses the tiebreaker metric to decide
   - If tiebreaker is `val_loss` → **lower is better**
   - Otherwise → **higher is better**

3. **Epoch Skipping**: Tracking starts after `skip_epochs_for_best` epochs
   - First N epochs are not considered
   - Avoids early random fluctuations

### Training Flow

```
Epoch 1-5:      SKIPPED (not tracked)

Epoch 6:        test_pm_acc=0.786, val_loss=0.6718
                → First tracked epoch
                → Save as best model (first time)

Epoch 7-19:     test_pm_acc < 0.786
                → Not better, don't update

Epoch 20:       test_pm_acc=0.786, val_loss=0.6650
                → Primary metric TIED (0.786 == 0.786)
                → Check tiebreaker: 0.6650 < 0.6718 ✓ BETTER
                → Update best model

Epoch 25:       test_pm_acc=0.857, val_loss=0.6700
                → Primary metric BETTER (0.857 > 0.786)
                → Update best model

Training End:   Load best model from epoch 25
                Evaluate on AC patients
                Save results with best model metrics
```

## Output and Results

### Console Output

During training, you'll see:

```
Epoch 6/30 - Training Loss: 0.5691 - Training acc: 0.789, ...
✓ Saved best model at epoch 6 | test_pm_acc=0.786

...

Epoch 20/30 - Training Loss: 0.4100 - Training acc: 0.900, ...
✓ Saved best model at epoch 20 | test_pm_acc=0.786

...

✓ Loaded best model from epoch 21
  Best metrics: {'test_pm_acc': 0.857, 'val_loss': 0.6700, ...}

✓ Evaluated best model on validation set (AC patients)
  Validation Loss: 0.6650
  Patient-wise Accuracy: 0.857
  Patient-wise AUC: 0.920
```

### Saved Files

1. **Model Checkpoint**: `exp_dir/fold_X/checkpoints/best_model.pth`
   - Contains: epoch number, model weights, metrics

2. **Results**: All results are saved based on best model evaluation
   - Updated with AC patient evaluation
   - Includes best_epoch information

## Usage Example

### Complete Configuration

```python
args = {
    # ... existing parameters ...
    
    # Best model selection (NEW)
    'save_best_model': True,
    'best_model_metric': 'test_pm_acc',         # Primary: patient-wise accuracy
    'best_model_tiebreaker': 'val_loss',        # Tiebreaker: validation loss
    'skip_epochs_for_best': 5,                  # Skip first 5 epochs
    
    # ... rest of parameters ...
}
```

### Running Training

```python
from traintest_without_GRL import train_without_GRL

# Training will automatically:
# 1. Skip first 5 epochs
# 2. Track best model based on test_pm_acc (primary)
# 3. Use val_loss for tie-breaking
# 4. Save best_model.pth checkpoint
# 5. Load and evaluate best model on AC patients
# 6. Return best_epoch in results

stats, val_loss, val_aucs, results_df, patient_results, \
    patient_metrics, train_aucs, train_loss, best_model_epoch = \
    train_without_GRL(audio_model_fold, train_loader, test_loader, args)

print(f"Best model found at epoch: {best_model_epoch+1}")
```

## New Helper Functions

Three new helper functions have been added to `traintest_without_GRL.py`:

### 1. `is_better_model(current_metrics, best_metrics, args)`

Compares two sets of metrics and returns True if current is better.

```python
# Internal logic
if primary_metric == 'val_loss':
    if current_primary < best_primary:
        return True
    elif current_primary == best_primary:  # Check tiebreaker
        return use_tiebreaker(...)
else:  # accuracy, AUC
    if current_primary > best_primary:
        return True
    elif current_primary == best_primary:  # Check tiebreaker
        return use_tiebreaker(...)
```

### 2. `save_best_model_checkpoint(audio_model, epoch_metrics, exp_dir, epoch)`

Saves model weights and metrics when a better model is found.

```python
# Creates: exp_dir/checkpoints/best_model.pth
# Contains: {
#     'epoch': epoch_number,
#     'model_state_dict': model_weights,
#     'metrics': epoch_metrics
# }
```

### 3. `load_best_model_and_evaluate(audio_model, exp_dir, val_loader, args, device)`

Loads the best model and evaluates it on the validation set (AC patients).

```python
# Returns: (stats, val_loss, results_df, patient_results, patient_metrics, best_epoch)
# Prints: Detailed evaluation metrics
```

## Backward Compatibility

If you don't want to use best model selection:

```python
args['save_best_model'] = False  # Disable feature
```

The code will work exactly as before:
- No checkpoints created
- Final model saved (not best model)
- `best_model_epoch = -1` returned

## Metric Names

The implementation supports tracking these metrics:

- `test_pm_acc` - Patient-wise accuracy on test set
- `test_pm_auc` - Patient-wise AUC on test set
- `test_acc` - Per-audio accuracy on test set
- `test_auc` - Per-audio AUC on test set
- `val_loss` - Validation loss
- `train_loss` - Training loss

## Cross-Validation Integration

In the cross-validation wrapper (`run_without_GRL_CV.ipynb`):

1. Each fold tracks its own best model
2. Best model is saved per fold in `fold_X/checkpoints/best_model.pth`
3. AC patient evaluation is done with best model
4. Results are saved per fold based on best model
5. Cross-fold statistics are aggregated from best models

## Troubleshooting

### Best model not being saved
- Check `save_best_model=True` in args
- Verify epoch number is >= `skip_epochs_for_best`
- Check that metrics are being calculated correctly

### Different results than expected
- Verify metric names match your setup
- Check if tie-breaking is working as expected
- Ensure `skip_epochs_for_best` is set correctly

### Memory issues
- Only model weights are saved (not optimizer), so should be lightweight
- Each fold overwrites the `best_model.pth` file
- Old checkpoints are automatically replaced

## Example Output Structure

```
exp_dir/
├── fold_1/
│   ├── checkpoints/
│   │   └── best_model.pth (best model weights)
│   ├── results_df.csv
│   ├── patient_results.csv
│   ├── patient_metrics.csv
│   ├── AC_results.csv (AC evaluation)
│   ├── AC_PW_results.csv
│   └── AC_patient_metrics.csv
├── fold_2/
│   └── ...
├── cv_metrics.csv (summary across folds)
└── cv_AC_metrics.csv (AC summary across folds)
```


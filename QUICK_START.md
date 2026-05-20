# Best Model Selection - Quick Start Guide

## 📌 What's New?

The training module now automatically:
1. **Tracks the best model** during training based on your chosen metric
2. **Breaks ties** using a secondary metric
3. **Skips early epochs** to avoid random fluctuations
4. **Evaluates AC patients** with the best model
5. **Saves all results** based on the best model

## 🚀 How to Enable (3 Steps)

### Step 1: Add Parameters to Your `args` Dictionary

```python
args = {
    # ... existing parameters ...
    
    # NEW: Best model selection
    'save_best_model': True,
    'best_model_metric': 'test_pm_acc',      # Primary metric
    'best_model_tiebreaker': 'val_loss',     # Tiebreaker metric
    'skip_epochs_for_best': 5,               # Skip first N epochs
}
```

### Step 2: Update Your Training Call

**Before:**
```python
stats, val_loss, val_aucs, results_df, patient_results, \
    patient_metrics, train_aucs, train_loss = \
    train_without_GRL(audio_model_fold, train_loader, test_loader, args)
```

**After:**
```python
stats, val_loss, val_aucs, results_df, patient_results, \
    patient_metrics, train_aucs, train_loss, best_model_epoch = \
    train_without_GRL(audio_model_fold, train_loader, test_loader, args)
```

### Step 3: Run Training

```python
# Your code will now automatically:
# 1. Skip first 5 epochs
# 2. Track best model based on test_pm_acc
# 3. Break ties with val_loss
# 4. Save best_model.pth checkpoint
# 5. Load and evaluate best model on AC patients
# 6. Save results based on best model

print(f"Best model found at epoch: {best_model_epoch+1}")
```

## 📊 What You'll See

**Console output during training:**

```
Epoch 6/30 - Training Loss: 0.5691 - Training acc: 0.789, ...
✓ Saved best model at epoch 6 | test_pm_acc=0.786

Epoch 20/30 - Training Loss: 0.4100 - Training acc: 0.900, ...
✓ Saved best model at epoch 20 | test_pm_acc=0.786

Epoch 25/30 - Training Loss: 0.3800 - Training acc: 0.920, ...
✓ Saved best model at epoch 25 | test_pm_acc=0.857

...

✓ Loaded best model from epoch 26
  Best metrics: {'test_pm_acc': 0.857, 'val_loss': 0.6700, ...}

✓ Evaluated best model on validation set (AC patients)
  Validation Loss: 0.6650
  Patient-wise Accuracy: 0.857
  Patient-wise AUC: 0.920
```

## 📁 Where Files Are Saved

```
exp_dir/
└── fold_X/
    ├── checkpoints/
    │   └── best_model.pth (best model weights)
    ├── results_df.csv (best model evaluation)
    ├── patient_results.csv
    ├── patient_metrics.csv
    ├── AC_results.csv (AC evaluation with best model)
    ├── AC_PW_results.csv
    └── AC_patient_metrics.csv
```

## ⚙️ Configuration Options

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `save_best_model` | False | Enable/disable feature |
| `best_model_metric` | - | Primary metric (e.g., 'test_pm_acc') |
| `best_model_tiebreaker` | - | Tiebreaker metric (e.g., 'val_loss') |
| `skip_epochs_for_best` | 5 | Skip first N epochs |

## 📝 Available Metrics

You can track any of these:
- `test_pm_acc` - Patient-wise accuracy
- `test_pm_auc` - Patient-wise AUC
- `test_acc` - Per-audio accuracy
- `test_auc` - Per-audio AUC
- `val_loss` - Validation loss
- `train_loss` - Training loss

## 🔄 How Tie-Breaking Works

If two epochs have the **same** `test_pm_acc`:
1. The code checks `val_loss` for both epochs
2. Selects the one with **lower** `val_loss`
3. Saves that as the best model

**Example:**
```
Epoch 15: test_pm_acc=0.800, val_loss=0.650
Epoch 22: test_pm_acc=0.800, val_loss=0.630  ← WINS (lower val_loss)
```

## ⚡ To Disable

If you don't want best model selection:

```python
args['save_best_model'] = False
```

Your code will work exactly as before.

## 📚 Full Documentation

See `BEST_MODEL_SELECTION_GUIDE.md` for:
- Detailed implementation guide
- Metric comparison logic
- Troubleshooting
- Example output structures

## ❓ FAQ

**Q: Will this change my current results?**
A: No, only if you enable `save_best_model=True`

**Q: How much extra storage does this use?**
A: Very little! Only model weights are saved (no optimizer state)

**Q: Can I use different metrics?**
A: Yes! Change `best_model_metric` and `best_model_tiebreaker` to any supported metric

**Q: What if there's no checkpoint file?**
A: The code falls back to the current model and returns `best_model_epoch=-1`

**Q: Does this work with AC evaluation?**
A: Yes! AC patients are automatically evaluated with the best model

---

**That's it! You're ready to use best model selection.**

For any issues, check `BEST_MODEL_SELECTION_GUIDE.md` or review the implementation in `src/traintest_without_GRL.py`

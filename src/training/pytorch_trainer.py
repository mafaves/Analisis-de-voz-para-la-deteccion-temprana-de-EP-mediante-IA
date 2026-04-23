import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PyTorchTrainer:
    """
    Trainer for PyTorch DL models.

    Handles:
    - Training loop with validation
    - Early stopping
    - Mixed precision training
    - Model checkpointing
    - Cross-validation support

    Example:
        >>> from src.models.pytorch import CNN2D
        >>> from src.training import PyTorchTrainer
        >>> trainer = PyTorchTrainer(model=CNN2D(), n_epochs=50)
        >>> results = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model,
        n_epochs=50,
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=32,
        loss='cross_entropy',
        optimizer='adam',
        scheduler='multistep',
        scheduler_params=None,
        early_stop_patience=10,
        monitor_metric='val_loss',
        use_amp=True,
        device=None
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params or {'milestones': [20, 40], 'gamma': 0.1}
        self.early_stop_patience = early_stop_patience
        self.monitor_metric = monitor_metric
        self.use_amp = use_amp

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.scaler = torch.amp.GradScaler() if use_amp else None

        self.best_metric = float('inf') if monitor_metric == 'val_loss' else -float('inf')
        self.early_stop_counter = 0

    def _create_optimizer(self):
        if self.optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )

    def _create_scheduler(self):
        if self.scheduler_name == 'multistep':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_params.get('milestones', [20, 40]),
                gamma=self.scheduler_params.get('gamma', 0.1)
            )
        elif self.scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.n_epochs
            )
        return None

    def _create_criterion(self):
        if self.loss == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.loss == 'bce':
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        for batch in train_loader:
            audio, labels, _ = batch
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                outputs = self.model(audio)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(loss.item(), audio.size(0))

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        return loss_meter.avg, accuracy

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        loss_meter = AverageMeter()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                audio, labels, _ = batch
                audio = audio.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(audio)
                loss = self.criterion(outputs, labels)

                loss_meter.update(loss.item(), audio.size(0))
                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        _, predicted = all_outputs.max(1)
        accuracy = predicted.eq(all_labels).sum().item() / all_labels.size(0)

        return loss_meter.avg, accuracy

    def train(
        self,
        train_loader,
        val_loader=None,
        save_dir=None,
        verbose=True
    ):
        """
        Train the model.

        Args:
            train_loader: PyTorch DataLoader for training data.
            val_loader: PyTorch DataLoader for validation data.
            save_dir: Directory to save results.
            verbose: Print progress.

        Returns:
            dict: Training history.
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(self.n_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                current_metric = val_loss if self.monitor_metric == 'val_loss' else val_acc

                if (self.monitor_metric == 'val_loss' and current_metric < self.best_metric - 1e-4) or \
                   (self.monitor_metric != 'val_loss' and current_metric > self.best_metric + 1e-4):
                    self.best_metric = current_metric
                    self.early_stop_counter = 0

                    if save_dir:
                        self.save_model(save_dir, 'best_model.pth')
                else:
                    self.early_stop_counter += 1

                if verbose:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}, "
                          f"Time: {time.time()-start_time:.1f}s")

                if self.early_stop_counter >= self.early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")

            if self.scheduler:
                self.scheduler.step()

        if save_dir:
            self.save_results(save_dir, history)

        return history

    def save_model(self, save_dir, filename):
        """Save model state dict."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, filename))

    def save_results(self, save_dir, history):
        """Save training history."""
        np.save(os.path.join(save_dir, 'history.npy'), history)

    def load_model(self, filepath):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
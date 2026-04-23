import torch.nn as nn


class AudioBaseModel(nn.Module):
    """
    Base class for audio classification models.

    Provides common interface for all PyTorch models.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        pass

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def init_weights(module):
    """Initialize model weights."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm1d, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
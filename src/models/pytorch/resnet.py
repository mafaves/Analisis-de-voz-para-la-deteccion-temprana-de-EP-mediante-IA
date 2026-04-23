import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetAttention(nn.Module):
    """
    ResNet50 with attention pooling for spectrogram classification.

    Input: (batch, 1, freq_bins, time_frames) - Mel spectrogram
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.2,
        dropatt_rate=0.2
    ):
        super().__init__()

        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = resnet50(weights=None)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()

        from .attention import AttentionPooling
        self.attention = AttentionPooling(2048, num_classes, dropout)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        batch, channels, freq, time = x.shape
        x = x.reshape(batch, channels, freq * time)
        x = x.permute(0, 2, 1)

        return self.attention(x)
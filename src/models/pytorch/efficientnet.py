import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EffNetAttention(nn.Module):
    """
    EfficientNet-B0/B1 with attention pooling for spectrogram classification.

    Input: (batch, 1, freq_bins, time_frames) - Mel spectrogram
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes=2,
        b=0,
        pretrained=True,
        dropout=0.2,
        use_efficientnetv2=False
    ):
        super().__init__()
        self.use_efficientnetv2 = use_efficientnetv2
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]

        if use_efficientnetv2:
            raise NotImplementedError("EfficientNetV2 not implemented yet")
        else:
            if pretrained:
                model_name = f'efficientnet-b{b}'
                self.effnet = EfficientNet.from_pretrained(model_name, in_channels=1)
            else:
                self.effnet = EfficientNet.from_name(f'efficientnet-b{b}', in_channels=1)

        self.effnet._fc = nn.Identity()

        from .attention import AttentionPooling
        self.attention = AttentionPooling(self.middim[b], num_classes, dropout)

    def forward(self, x):
        x = self.effnet.extract_features(x)

        batch, channels, freq, time = x.shape
        x = x.reshape(batch, channels, freq * time)
        x = x.permute(0, 2, 1)

        return self.attention(x)
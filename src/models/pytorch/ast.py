import torch.nn as nn
import timm


class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer (AST) for audio classification.

    Uses pretrained AST model from timm with spectrogram input.

    Input: (batch, 1, freq_bins, time_frames) - Mel spectrogram
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes=2,
        model_name='vit_base_patch16_384',
        pretrained=True,
        dropout=0.2,
        freeze_patch_embeddings=False
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=1
        )

        self.freeze_patch_embeddings = freeze_patch_embeddings
        if freeze_patch_embeddings:
            self.freeze_patch_embed()

    def freeze_patch_embed(self):
        """Freeze patch embedding layers."""
        for name, param in self.model.named_parameters():
            if 'patch_embed' in name or 'cls_token' in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
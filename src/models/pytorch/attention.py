import torch.nn as nn
import torch


class AttentionPooling(nn.Module):
    """
    Single-head attention pooling for audio classification.
    """

    def __init__(self, input_dim, num_classes=2, dropout=0.2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.classifier = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        context = self.dropout(context)
        return self.classifier(context)


class MHeadAttention(nn.Module):
    """
    Multi-head attention pooling for audio classification.
    """

    def __init__(
        self,
        input_dim,
        num_classes=2,
        num_heads=4,
        dropout=0.2
    ):
        super().__init__()
        self.num_heads = num_heads

        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Tanh(),
                nn.Linear(input_dim, 1)
            )
            for _ in range(num_heads)
        ])

        self.head_weight = nn.Parameter(torch.ones(num_heads) / num_heads)

        self.classifier = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = []

        for attn in self.attention:
            attn_weights = torch.softmax(attn(x), dim=1)
            context = torch.sum(attn_weights * x, dim=1)
            head_outputs.append(context)

        head_outputs = torch.stack(head_outputs, dim=0)
        aggregated = torch.sum(self.head_weight * head_outputs, dim=0)

        aggregated = self.dropout(aggregated)
        return self.classifier(aggregated)


class DropAttention(nn.Module):
    """
    DropAttention as described in paper with post-dropout normalization.
    """

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, att_weights):
        if not self.training or self.p == 0:
            return att_weights

        mask = torch.rand(
            att_weights.size(0), 1, att_weights.size(2),
            device=att_weights.device
        ) > self.p

        masked_att = att_weights * mask.float()
        normalized_att = masked_att / (masked_att.sum(dim=2, keepdim=True) + 1e-7)

        return normalized_att


class MeanPooling(nn.Module):
    """
    Mean pooling for audio classification.
    """

    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        context = torch.mean(x, dim=1)
        return self.classifier(context)
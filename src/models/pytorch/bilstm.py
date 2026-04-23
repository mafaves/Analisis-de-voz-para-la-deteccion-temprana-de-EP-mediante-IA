import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for audio classification.

    Input: (batch, time_steps, features) - e.g., flattened spectrogram
    Output: (batch, num_classes)

    Supports attention mechanism for better temporal modeling.
    """

    def __init__(
        self,
        input_dim,
        num_classes=2,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        use_attention=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if use_attention:
            self.attention = nn.Linear(lstm_output_dim, 1)

        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Expected 2 or 3 dims, got {x.dim()}")

        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
        else:
            context = lstm_out[:, -1, :]

        context = self.dropout(context)
        output = self.fc(context)

        return output


class BiLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with multi-head attention for audio classification.
    """

    def __init__(
        self,
        input_dim,
        num_classes=2,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = hidden_dim * 2

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            pass

        lstm_out, _ = self.lstm(x)

        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out[:, -1, :]

        attn_out = self.dropout(attn_out)
        output = self.fc(attn_out)

        return output
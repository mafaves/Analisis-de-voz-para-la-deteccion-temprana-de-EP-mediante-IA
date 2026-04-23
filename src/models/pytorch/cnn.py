import torch.nn as nn
import torch


class CNN1D(nn.Module):
    """
    1D CNN for raw audio waveform classification.

    Input: (batch, 1, time_samples)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes=2,
        input_channels=1,
        hidden_channels=[32, 64, 128],
        kernel_size=5,
        stride=2,
        pool_size=2,
        dropout=0.3
    ):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        in_ch = input_channels

        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class CNN2D(nn.Module):
    """
    2D CNN for spectrogram classification.

    Input: (batch, 1, freq_bins, time_frames) - e.g., Mel spectrogram
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes=2,
        input_channels=1,
        hidden_channels=[32, 64, 128],
        kernel_size=(3, 3),
        stride=(1, 1),
        pool_size=(2, 2),
        dropout=0.3
    ):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        in_ch = input_channels

        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                nn.Dropout2d(dropout)
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class CNNLSTM(nn.Module):
    """
    CNN + LSTM hybrid for spectrogram classification.

    Uses CNN for feature extraction, then LSTM for temporal modeling.
    """

    def __init__(
        self,
        num_classes=2,
        cnn_hidden=[32, 64],
        lstm_hidden=128,
        num_lstm_layers=2,
        dropout=0.3
    ):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_hidden[0], 3, padding=1),
            nn.BatchNorm2d(cnn_hidden[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_hidden[0], cnn_hidden[1], 3, padding=1),
            nn.BatchNorm2d(cnn_hidden[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_hidden[1],
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, freq, time = x.size()
        x = x.permute(0, 2, 1, 3).reshape(batch, time, channels * freq)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
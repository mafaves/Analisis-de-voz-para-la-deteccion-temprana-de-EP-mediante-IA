import torch
from torch.utils.data import Dataset
import librosa
import numpy as np


def calculate_mean_std(dataset, batch_size=32):
    """
    Calculate mean and std of a dataset for normalization.

    Args:
        dataset: PyTorch Dataset object.
        batch_size (int): Batch size for processing.

    Returns:
        tuple: (mean, std)
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    means, stds = [], []

    for mel_tensor, _, _ in loader:
        means.append(torch.mean(mel_tensor).item())
        stds.append(torch.std(mel_tensor).item())

    return np.mean(means), np.mean(stds)


class ParkinsonAudioDataset(Dataset):
    """
    PyTorch Dataset for Parkinson's voice analysis.

    Converts raw audio to Mel spectrogram on-the-fly.

    Args:
        audio_segments (np.array): Raw audio waveforms.
        labels (np.array): Labels for each segment.
        patient_ids (np.array): Patient IDs for each segment.
        n_mels (int): Number of mel bands.
        sr (int): Sampling rate.
        hop_length (int): Hop length for STFT.
        n_fft (int): FFT window size.
        normalize (bool): Apply normalization.
        norm_mean (float): Mean for normalization.
        norm_std (float): Std for normalization.
        augment (bool): Apply data augmentation.
    """

    def __init__(
        self,
        audio_segments,
        labels,
        patient_ids,
        n_mels=128,
        sr=16000,
        hop_length=256,
        n_fft=512,
        normalize=False,
        norm_mean=0,
        norm_std=1,
        augment=False,
        freq_mask=20,
        time_mask=200,
        augmentation_rate=0.5
    ):
        self.audio_segments = audio_segments
        self.labels = labels
        self.patient_ids = patient_ids
        self.n_mels = n_mels
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.augment = augment
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.augmentation_rate = augmentation_rate

        self.label_to_indices = self._build_label_indices()

    def _build_label_indices(self):
        """Build mapping from label to indices for mixup augmentation."""
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        """Get one sample."""
        audio = self.audio_segments[idx]
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        mel = librosa.feature.melspectrogram(
            audio, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32)

        if self.normalize:
            mel_tensor = (mel_tensor - self.norm_mean) / self.norm_std

        if self.augment and np.random.random() < self.augmentation_rate:
            mel_tensor = self._apply_augmentation(mel_tensor)

        return mel_tensor, torch.tensor(label, dtype=torch.long), patient_id

    def _apply_augmentation(self, mel_tensor):
        """Apply SpecAugment-style augmentation."""
        freq_mask = torchaudio.transforms.FrequencyMasking(self.freq_mask)
        time_mask = torchaudio.transforms.TimeMasking(self.time_mask)

        mel_tensor = mel_tensor.unsqueeze(0)
        mel_tensor = freq_mask(mel_tensor)
        mel_tensor = time_mask(mel_tensor)
        mel_tensor = mel_tensor.squeeze(0)

        if np.random.random() < 0.5:
            noise_std = np.random.uniform(0, 0.05)
            mel_tensor = mel_tensor + torch.randn_like(mel_tensor) * noise_std

        return mel_tensor
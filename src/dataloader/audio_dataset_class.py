import torch
import torchaudio
import librosa
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


def calculate_mean_std(dataset, batch_size=32, GRL = False):
	
	"""
	Calculate the mean and standard deviation of a dataset.

	This function computes the mean and standard deviation of the input data, typically used for normalization during training. It processes the data in batches to handle large datasets efficiently.

	Args:
	- dataset: PyTorch Dataset object containing the data to be analyzed. Each data sample is expected to return a tensor as the first output.
	- batch_size: The number of samples to process in each batch (default is 32).

	Returns:
	- dataset_mean: The overall mean of the dataset (computed across all batches).
	- dataset_std: The overall standard deviation of the dataset (computed across all batches).
	"""
	mean = []
	std = []

	# Create a DataLoader for the dataset to loop through it
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	for mel_tensor, _ , _ in loader:
		cur_mean = torch.mean(mel_tensor)
		cur_std = torch.std(mel_tensor)
		mean.append(cur_mean.item())
		std.append(cur_std.item())

	# Calculate the overall mean and std for the dataset
	dataset_mean = np.mean(mean)
	dataset_std = np.mean(std)

	return dataset_mean, dataset_std


class ParkinsonAudioDataset_without_GRL(Dataset):
	def __init__(self, audio_segments, labels, patient_ids, n_mels=128, sr=16000, hop_length=256, n_fft=512, win_length = 400, normalize = False, transpose = False, norm_mean=0, norm_std=0.5, augment=False, mixup_rate = 0.5, freq_mask = 20, time_mask = 200, std = 0.5, noise = True, augmentation_rate = 0.5):
		self.audio_segments = audio_segments
		self.labels = labels
		self.patient_ids = patient_ids
		self.n_mels = n_mels
		self.sr = sr
		self.hop_length = hop_length
		self.n_fft = n_fft
		self.win_length = win_length
		self.normalize = normalize
		self.transpose = transpose
		self.norm_mean = norm_mean
		self.norm_std = norm_std
		self.augment = augment
		self.mixup_rate = mixup_rate
		self.freq_mask = freq_mask
		self.time_mask = time_mask
		self.std = std
		self.noise = noise
		self.augmentation_rate = augmentation_rate


		# Map each label to the indices of its samples for mix-up
		self.label_to_indices = self._build_label_indices()


	def __len__(self):
		return len(self.audio_segments)

	def _build_label_indices(self):
		"""
		Builds a dictionary mapping each label to a list of sample indices.
		This is used to ensure mix-up is only done within the same label.
		"""
		label_to_indices = {}
		for idx, label in enumerate(self.labels):
			if label not in label_to_indices:
				label_to_indices[label] = []
			label_to_indices[label].append(idx)
		return label_to_indices

	def _apply_masking(self, mel_tensor):
		"""
		Apply data augmentation techniques (frequency and time masking) to the Mel-spectrogram.

		Parameters:
		- mel_tensor: Input Mel-spectrogram tensor.

		Returns:
		- Augmented Mel-spectrogram tensor.
		"""
		# Example augmentations
		# SpecAugment: Frequency and Time Masking
		freqm = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask)  # Mask up to X frequency bins
		timem = torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask)	   # Mask up to X time frames

		mel_tensor = mel_tensor.unsqueeze(0)  # Add batch dimension for compatibility
		mel_tensor = freqm(mel_tensor)
		mel_tensor = timem(mel_tensor)
		mel_tensor = mel_tensor.squeeze(0)  # Remove batch dimension

		if self.noise:
			# Use Gaussian noise instead of uniform noise
			noise_std = np.random.uniform(0, 0.05)  # Adjust max std as needed
			mel_tensor = mel_tensor + torch.randn_like(mel_tensor) * noise_std
			#mel_tensor = mel_tensor + torch.rand(mel_tensor.shape[0], mel_tensor.shape[1]) * np.random.rand() / 10 #Random Noise Addition
			mel_tensor = torch.roll(mel_tensor, np.random.randint(-10, 10), 1) # Rolling the Spectrogram

		return mel_tensor

	def _apply_mixup(self, audio_input, label):
		"""
		Performs mix-up augmentation by combining the current spectrogram
		with another spectrogram of the same label.

		Parameters:
		- mel_tensor: The current Mel-spectrogram tensor.
		- label: The label of the current sample.

		Returns:
		- Mixed Mel-spectrogram.
		"""

		# Find another sample with the same label
		candidates = self.label_to_indices[label]
		mix_idx = random.choice(candidates)

		# Load the other spectrogram
		mix_audio = self.audio_segments[mix_idx]

		if audio_input.shape[0] != mix_audio.shape[0]:
			if audio_input.shape[0] > mix_audio.shape[0]:
				# padding
				temp_wav = torch.zeros(1, audio_input.shape[0])
				temp_wav[0, 0:mix_audio.shape[0]] = mix_audio
				mix_audio = temp_wav
			else:
				# cutting
				mix_audio = mix_audio[0, 0:audio_input.shape[1]]

		mix_lambda = np.random.beta(10, 10)

		mix_waveform = mix_lambda * audio_input + (1 - mix_lambda) * mix_audio

		mix_mel = librosa.feature.melspectrogram(y = mix_waveform, sr=self.sr, n_fft=self.n_fft,
												 hop_length=self.hop_length, n_mels=self.n_mels, win_length=self.win_length)
		mix_mel_db = librosa.power_to_db(mix_mel, ref=np.max)
		mix_mel_tensor = torch.tensor(mix_mel_db, dtype=torch.float32)

		return mix_mel_tensor
	
	def _apply_augmentation(self, audio, label, mel_tensor):

		random_number = random.random()
		if random_number < self.mixup_rate:
			# Mix-up augmentation
			mel_tensor = self._apply_mixup(audio, label)
		else:
			# SpecAugment: Frequency and Time Masking
			mel_tensor = self._apply_masking(mel_tensor)
				
		return mel_tensor



	def __getitem__(self, idx):
		# Get audio segment and label
		audio = self.audio_segments[idx]
		label = self.labels[idx]
		patient_id = self.patient_ids[idx]

		# Convert audio to Mel-spectrogram
		mel = librosa.feature.melspectrogram(y = audio, sr=self.sr, n_fft=self.n_fft,
											 hop_length=self.hop_length, n_mels=self.n_mels, win_length=self.win_length , window = 'hann', fmax = self.sr / 2.0)
		mel_db = librosa.power_to_db(mel, ref=np.max)

		# Convert to PyTorch tensor and add channel dimension
		mel_tensor = torch.tensor(mel_db, dtype=torch.float32)

		# Conditional Data Augmentation
		if self.augment:
			random_number = random.random()

			if random_number < self.augmentation_rate:
				mel_tensor = self._apply_augmentation(audio, label, mel_tensor)

		if self.normalize and self.std == 1:
			# Normalize the spectrogram to have mean=0 and std=1 (ResNet, EfficientNet...)
			mel_tensor = (mel_tensor - self.norm_mean) / (self.norm_std)

		if self.normalize and self.std == 0.5:
			# Normalize the spectrogram to have mean=0 and std=0.5 (AST)
			mel_tensor = (mel_tensor - self.norm_mean) / (self.norm_std*2)

		if self.transpose:
			# Transpose the dimensions: (batch_size, time_frames, frequency_bins)
			mel_tensor = mel_tensor.permute(1, 0)  # [frequency_bins, time_frames] -> [time_frames, frequency_bins]

		# Convert label to tensor
		label_tensor = torch.tensor(label, dtype=torch.long)

		return mel_tensor, label_tensor, patient_id

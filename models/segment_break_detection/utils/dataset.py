import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
from functools import lru_cache
import numpy as np


class BreakDataset(Dataset):
    def __init__(self, audio_dir: str, label_dir: str, interval_width: int = 20,
                 img_width: int = 64, img_height: int = 64,
                 cache_size: int = 10, debug: bool = False):
        """
        Args:
            audio_dir: Directory containing audio files
            label_dir: Directory containing label files
            interval_width: Width of intervals in ms
            img_width: Width of each spectrogram slice image
            img_height: Height of each spectrogram slice image
            cache_size: Number of files to keep in memory cache
            debug: Enable debug printing
        """
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.interval_width = interval_width
        self.img_width = img_width
        self.img_height = img_height
        self.debug = debug

        # Precompute file indices
        self.file_indices = []
        for audio_file in os.listdir(audio_dir):
            try:
                file_idx = int(audio_file.split('.')[0])
                label_path = os.path.join(label_dir, f"{file_idx}.csv")

                if os.path.exists(label_path):
                    self.file_indices.append(file_idx)
                    if debug:
                        print(f"Indexed file: {file_idx}")
                elif debug:
                    print(f"Skipping file {file_idx} - no corresponding label")
            except (ValueError, IndexError):
                if debug:
                    print(f"Skipping invalid filename: {audio_file}")

        self.length = len(self.file_indices)
        if debug:
            print(f"Total valid files: {self.length}")

    @lru_cache(maxsize=10)
    def _load_file_data(self, file_idx: int):
        """
        Cached method to load and process a single file's data

        Args:
            file_idx: Index of the file to load

        Returns:
            Tuple of (image_slices, labels)
        """
        # Construct file paths
        audio_file = f"{file_idx}.mp3"  # or .wav
        audio_path = os.path.join(self.audio_dir, audio_file)
        label_path = os.path.join(self.label_dir, f"{file_idx}.csv")

        # Load labels
        df = pd.read_csv(label_path)
        labels = torch.tensor(df['break'].values, dtype=torch.bool)

        # Load audio and compute image slices
        waveform, sr = torchaudio.load(audio_path)
        image_slices = self._compute_image_slices(waveform, sr)

        # Truncate to last break + 2
        last_break = labels.nonzero(as_tuple=True)[0][-1]
        image_slices = image_slices[:last_break + 2]
        labels = labels[:last_break + 2]

        if self.debug:
            print(f"Loaded file {file_idx}: Images shape {image_slices.shape}, Labels shape {labels.shape}")

        return image_slices, labels

    def _compute_image_slices(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

        # Compute mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            hop_length=int(sr * self.interval_width / 1000),
            n_mels=self.img_height  # Set n_mels to match desired image height
        )(waveform)

        mel_spec = mel_spec.squeeze(0)

        # Convert to log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        # Convert to image format [num_frames, img_width, img_height]
        transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.Normalize(0, 1)  # Ensure values are in [0,1] for grayscale
        ])

        # Reshape to [num_frames, img_width, img_height] and apply transform
        num_frames = mel_spec.shape[1]
        image_slices = torch.zeros(num_frames, self.img_width, self.img_height)

        for i in range(num_frames):
            slice_spec = mel_spec[:, i:i + 1]  # Take one time slice
            slice_spec = slice_spec.unsqueeze(0)  # Add batch dimension for transform
            image_slices[i] = transform(slice_spec).squeeze(0)

        return image_slices

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Lazily load data for the requested index

        Returns:
            Tuple of (file_idx, image_slices, labels)
        """
        file_idx = self.file_indices[idx]
        image_slices, labels = self._load_file_data(file_idx)
        return file_idx, image_slices, labels

    def clear_cache(self):
        """
        Clear the LRU cache to free up memory
        """
        self._load_file_data.cache_clear()

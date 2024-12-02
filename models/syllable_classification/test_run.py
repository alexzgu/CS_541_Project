import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import hashlib
import pickle
import logging
import subprocess

# load in data with dataloader
data_dir = '../../data'
syllable_directory = f'{data_dir}/clean/syllables'
audio_clip_directory = f'{syllable_directory}/clips'  # this contains indexed .mp3 files, each of which is a song
clip_index_file = f'{syllable_directory}/segment_index.csv'
# the clip index file contains the following columns:
# index: the index of the corresponding audio clip
# token: the token (i.e., label) corresponding to the clip (string)

# I want a custom dataset class such that each sample pairs the audio clip with its corresponding token
# I will then use a DataLoader to load in the data

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CachedFeatureDataset(Dataset):
    def __init__(self, clip_index_file: str, audio_clip_directory: str,
                 cache_directory: str, max_length=16000):
        self.clip_index = pd.read_csv(clip_index_file)
        self.audio_clip_directory = audio_clip_directory
        self.cache_directory = cache_directory
        self.max_length = max_length

        # Ensure cache directory exists
        os.makedirs(cache_directory, exist_ok=True)

        # Bundle for feature extraction
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.wav2vec_model = self.bundle.get_model()

        # Create token to label mapping
        self.token_to_label = {token: idx for idx, token in enumerate(self.clip_index['token'].unique())}

        # Validate audio files
        self._validate_audio_files()

    def _validate_audio_files(self):
        """
        Validate that all audio files exist and are readable
        """
        invalid_files = []
        for idx in range(len(self.clip_index)):
            clip_name = f"{self.clip_index.iloc[idx]['index']}.mp3"
            clip_path = os.path.join(self.audio_clip_directory, clip_name)

            if not os.path.exists(clip_path):
                invalid_files.append(clip_path)
                logger.warning(f"File not found: {clip_path}")

        if invalid_files:
            logger.error(f"Found {len(invalid_files)} invalid audio files")
            raise FileNotFoundError(f"Missing audio files: {invalid_files[:10]}...")

    def _generate_cache_key(self, clip_path):
        # Create a unique hash for the audio file
        try:
            with open(clip_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error generating cache key for {clip_path}: {e}")
            raise

    def _extract_and_cache_features(self, clip_path):
        # Check if features are already cached
        try:
            cache_key = self._generate_cache_key(clip_path)
            cache_file = os.path.join(self.cache_directory, f"{cache_key}_features.pkl")

            # If cached, load from file
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

            # Load audio file with error handling
            try:
                waveform, sample_rate = torchaudio.load(clip_path)
            except Exception as e:
                logger.error(f"Failed to load audio file {clip_path}: {e}")
                # Return a dummy tensor to prevent training from stopping
                return torch.zeros(768)  # Match Wav2Vec2 feature dimension

            # Ensure mono channel and correct shape
            waveform = waveform.mean(dim=0) if waveform.ndim > 1 else waveform.squeeze()

            # Resample if necessary
            if sample_rate != self.bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)

            # Pad or truncate
            if len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.max_length - len(waveform)))

            # Ensure 2D input (batch, time)
            waveform = waveform.unsqueeze(0)

            # Extract features
            with torch.no_grad():
                features, _ = self.wav2vec_model.extract_features(waveform)
                pooled_features = features[-1].mean(dim=1).squeeze()

            # Cache features
            with open(cache_file, 'wb') as f:
                pickle.dump(pooled_features, f)

            return pooled_features

        except Exception as e:
            logger.error(f"Unexpected error processing {clip_path}: {e}")
            # Return a dummy tensor to prevent training from stopping
            return torch.zeros(768)  # Match Wav2Vec2 feature dimension

    def __len__(self):
        return len(self.clip_index)

    def __getitem__(self, idx):
        clip_name = f"{self.clip_index.iloc[idx]['index']}.mp3"
        clip_path = os.path.join(self.audio_clip_directory, clip_name)

        # Extract or load cached features
        features = self._extract_and_cache_features(clip_path)

        # Get token label
        token = self.clip_index.iloc[idx]['token']
        label = self.token_to_label[token]

        return features, label


def train_model(clip_index_file, audio_clip_directory, cache_directory):
    # Prepare dataset with feature caching
    syllable_dataset = CachedFeatureDataset(
        clip_index_file,
        audio_clip_directory,
        cache_directory
    )
    num_classes = len(syllable_dataset.token_to_label)

    # Create DataLoader
    syllable_dataloader = DataLoader(
        syllable_dataset,
        batch_size=32,  # Increased batch size due to pre-extracted features
        shuffle=True,
        num_workers=4,
        drop_last=True  # Drop last incomplete batch
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SimplifiedSyllableClassifier(num_classes).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, (inputs, labels) in enumerate(syllable_dataloader):
            # Move to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Logging
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch + 1}/{num_epochs}], "
                            f"Batch [{batch_idx}/{len(syllable_dataloader)}], "
                            f"Loss: {loss.item():.4f}")

        # Epoch summary
        logger.info(f"Epoch {epoch + 1} average loss: {total_loss / batch_count:.4f}")

    # Save model
    torch.save(model.state_dict(), "syllable_classifier_cached.pth")

    return model


class SimplifiedSyllableClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# Usage
data_dir = '../../data'
syllable_directory = f'{data_dir}/clean/syllables'
audio_clip_directory = f'{syllable_directory}/clips'
clip_index_file = f'{syllable_directory}/segment_index.csv'
cache_directory = f'{data_dir}/feature_cache'  # Specify your cache directory

# Run training
model = train_model(clip_index_file, audio_clip_directory, cache_directory)

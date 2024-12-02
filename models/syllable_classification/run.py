import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

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


class SyllableDataset(Dataset):
    def __init__(self, clip_index_file: str, audio_clip_directory: str, max_length=16000):
        self.clip_index = pd.read_csv(clip_index_file)
        self.audio_clip_directory = audio_clip_directory
        self.max_length = max_length

        # Use Wav2Vec2 ASR bundle
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        # Create a mapping of unique tokens to integer labels
        self.token_to_label = {token: idx for idx, token in enumerate(self.clip_index['token'].unique())}

    def _pad_or_truncate(self, audio):
        if len(audio) > self.max_length:
            return audio[:self.max_length]
        else:
            return np.pad(audio, (0, self.max_length - len(audio)), mode='constant')

    def __len__(self):
        return len(self.clip_index)

    def __getitem__(self, idx):
        clip_name = f"{self.clip_index.iloc[idx]['index']}.mp3"
        clip_path = os.path.join(self.audio_clip_directory, clip_name)

        # Load audio file
        waveform, sample_rate = torchaudio.load(clip_path)

        # Resample if necessary
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)

        # Ensure correct shape and length
        waveform = waveform.squeeze(0)  # Remove channel dimension if needed
        waveform = torch.tensor(self._pad_or_truncate(waveform.numpy()))

        # Get token label
        token = self.clip_index.iloc[idx]['token']
        label = self.token_to_label[token]

        return waveform, label


class SyllableClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SyllableClassifier, self).__init__()

        # Load pre-trained Wav2Vec2 model
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.wav2vec_model = bundle.get_model()

        # Freeze Wav2Vec2 layers
        for param in self.wav2vec_model.parameters():
            param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # Wav2Vec2-base has 768 features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features using Wav2Vec2
        with torch.no_grad():
            # Extract features from the model
            features, _ = self.wav2vec_model.extract_features(x)

            # Use the last layer's features
            last_layer_features = features[-1]

            # Global average pooling
            pooled_features = last_layer_features.mean(dim=1)

        # Classify
        return self.classifier(pooled_features)


def train_model(clip_index_file, audio_clip_directory):
    # Prepare dataset
    syllable_dataset = SyllableDataset(clip_index_file, audio_clip_directory)
    num_classes = len(syllable_dataset.token_to_label)

    # Create DataLoader
    syllable_dataloader = DataLoader(
        syllable_dataset,
        batch_size=16,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SyllableClassifier(num_classes).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),  # Only train classification layers
        lr=1e-4,
        weight_decay=1e-5
    )

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

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

            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Batch [{batch_idx}/{len(syllable_dataloader)}], "
                      f"Loss: {loss.item():.4f}")

        # Epoch summary
        print(f"Epoch {epoch + 1} average loss: {total_loss / len(syllable_dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), "syllable_classifier_wav2vec_torchaudio.pth")

    return model


# Run training
model = train_model(clip_index_file, audio_clip_directory)
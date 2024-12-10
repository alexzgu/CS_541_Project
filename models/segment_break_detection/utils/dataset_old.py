import torch
import torchaudio
from torch.utils.data import Dataset
import os
import pandas as pd

class BreakDataset(Dataset):
    def __init__(self, audio_dir: str, label_dir: str, interval_width: int = 20, debug: bool = False):
        """
        Args:
        audio_dir: all files in audio_dir are audio files with names of the form
        "{INTEGER}.{mp3, wav, etc.}"
        label_dir: all files in label_dir are csv files with names of the form
        "{INTEGER}.csv", where each csv file has columns
        index (non-negative int), start (non-negative float), end (non-negative float), break (bool)
        interval_width: width of the fixed intervals (in ms) used to compute the mel spectrogram.
        Default is 20 ms.
        """

        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.interval_width = interval_width
        self.length = 0
        self.data = []
        for audio_file in os.listdir(audio_dir):

            # because the indexing is based on filename (int), where some numbers are skipped
            file_idx = int(audio_file.split('.')[0])
            if debug:
                print(f"Processing file {file_idx}")

            # check if the corresponding label file exists; if not, skip
            label_path = os.path.join(label_dir, f"{file_idx}.csv")
            if not os.path.exists(label_path):
                if debug:
                    print(f"Skipping file {file_idx} because {label_path} does not exist")
                continue

            # load labels
            df = pd.read_csv(label_path)
            labels = torch.tensor(df['break'].values, dtype=torch.bool)

            waveform, sr = torchaudio.load(os.path.join(audio_dir, audio_file))
            mel_spec = self._compute_spectrogram(waveform, sr)

            # Truncate to last break + 2
            last_break = labels.nonzero(as_tuple=True)[0][-1]
            mel_spec = mel_spec[:, :last_break + 2]
            labels = labels[:last_break + 2]

            self.data.append((file_idx, mel_spec, labels))
            self.length += 1
            print(f"Index: {file_idx}, Mel shape: {mel_spec.shape}, Labels shape: {labels.shape}")
            if mel_spec.shape[1] != labels.shape[0]:
                print(f"WARNING: File {file_idx} has mismatched shapes {mel_spec.shape} and {labels.shape}")
        if debug:
            print(f"Successfully loaded {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]

    def _compute_spectrogram(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        # load audio and convert to spectrogram
        waveform = waveform.mean(dim=0, keepdim=True)  # go from 2 to 1 channel

        # Compute mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            hop_length=int(sr * self.interval_width / 1000),
            n_mels=32  # NOTE: this can be adjusted to modify the resolution of the spectrogram
        )(waveform)

        mel_spec = mel_spec.squeeze(0)  # go from [1, n_mels, n_frames] to [n_mels, n_frames]
        # where each n_frame is a {inter_width} ms interval

        # Add normalization
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        return mel_spec

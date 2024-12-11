import torch
import torchaudio
import pandas as pd
import jaconv  # for handling Japanese text conversions
from pathlib import Path
from typing import Dict, List, Tuple


class JapaneseSyllableProcessor:
    def __init__(self,
                 sample_rate: int = 16000,
                 frame_length: float = 0.025,  # 25ms frames
                 frame_shift: float = 0.010):  # 10ms shift
        """
        Processor for Japanese syllable-level audio analysis

        Args:
            sample_rate: Target audio sample rate
            frame_length: Analysis frame length in seconds
            frame_shift: Frame shift in seconds
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift

        # Basic hiragana syllable set (can be customized)
        self.basic_syllables = {
            'あ', 'い', 'う', 'え', 'お',
            'か', 'き', 'く', 'け', 'こ',
            'さ', 'し', 'す', 'せ', 'そ',
            'た', 'ち', 'つ', 'て', 'と',
            'な', 'に', 'ぬ', 'ね', 'の',
            'は', 'ひ', 'ふ', 'へ', 'ほ',
            'ま', 'み', 'む', 'め', 'も',
            'や', 'ゆ', 'よ',
            'ら', 'り', 'る', 'れ', 'ろ',
            'わ', 'を', 'ん',
            'が', 'ぎ', 'ぐ', 'げ', 'ご',
            'ざ', 'じ', 'ず', 'ぜ', 'ぞ',
            'だ', 'ぢ', 'づ', 'で', 'ど',
            'ば', 'び', 'ぶ', 'べ', 'ぼ',
            'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ',
            'きゃ', 'きゅ', 'きょ',
            'しゃ', 'しゅ', 'しょ',
            'ちゃ', 'ちゅ', 'ちょ',
            'にゃ', 'にゅ', 'にょ',
            'ひゃ', 'ひゅ', 'ひょ',
            'みゃ', 'みゅ', 'みょ',
            'りゃ', 'りゅ', 'りょ',
            'ぎゃ', 'ぎゅ', 'ぎょ',
            'じゃ', 'じゅ', 'じょ',
            'びゃ', 'びゅ', 'びょ',
            'ぴゃ', 'ぴゅ', 'ぴょ',
            'でぃ', 'ふぁ', 'ふぃ', 'ふぇ', 'ふぉ',
            '<silence>'
        }

        # Syllable to index mapping
        self.syllable_to_idx = {syl: idx for idx, syl in enumerate(self.basic_syllables)}
        self.idx_to_syllable = {idx: syl for syl, idx in self.syllable_to_idx.items()}

        # Audio preprocessing transforms
        self.audio_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=80,
            n_fft=int(frame_length * sample_rate),
            hop_length=int(frame_shift * sample_rate)
        )

        self.normalize = torchaudio.transforms.AmplitudeToDB()

    def load_and_preprocess_audio(self,
                                  audio_path: str,
                                  start_time: float = None,
                                  end_time: float = None) -> torch.Tensor:
        """
        Load and preprocess audio segment

        Args:
            audio_path: Path to MP3 file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds

        Returns:
            Preprocessed audio features
        """
        # Load audio
        waveform, orig_sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        # Extract segment if times provided
        if start_time is not None and end_time is not None:
            start_frame = int(start_time * self.sample_rate)
            end_frame = int(end_time * self.sample_rate)
            waveform = waveform[:, start_frame:end_frame]

        # Convert to mel spectrogram
        mel_spec = self.audio_transform(waveform)

        # Apply log-scale and normalization
        mel_spec = self.normalize(mel_spec)

        return mel_spec


class JapaneseSyllableDataset:
    def __init__(self,
                 audio_path: str,
                 annotation_csv: str,
                 processor: JapaneseSyllableProcessor):
        """
        Dataset for Japanese syllable-level analysis

        Args:
            audio_path: Path to MP3 file
            annotation_csv: Path to CSV with timing annotations
            processor: Syllable processor instance
        """
        self.audio_path = Path(audio_path)
        self.processor = processor

        # Load and process annotations
        self.annotations = self._load_annotations(annotation_csv)

    def _load_annotations(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess annotation CSV

        Args:
            csv_path: Path to annotation CSV

        Returns:
            Processed DataFrame
        """
        df = pd.read_csv(csv_path)

        # Assuming CSV has columns: token, start_time, end_time
        # Convert times to float if needed
        df['start_time'] = pd.to_numeric(df['start_time'])
        df['end_time'] = pd.to_numeric(df['end_time'])

        # Filter for valid syllables
        df = df[df['token'].isin(self.processor.basic_syllables)]

        return df

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a single syllable segment with its features

        Returns:
            Dict containing audio features and syllable info
        """
        row = self.annotations.iloc[idx]

        # Extract audio segment
        features = self.processor.load_and_preprocess_audio(
            self.audio_path,
            row['start_time'],
            row['end_time']
        )

        # Convert syllable to index
        syllable_idx = self.processor.syllable_to_idx[row['token']]

        return {
            'features': features,
            'syllable_idx': torch.tensor(syllable_idx, dtype=torch.long),
            'start_time': row['start_time'],
            'end_time': row['end_time']
        }


def prepare_training_data(audio_path: str,
                          annotation_csv: str,
                          batch_size: int = 32) -> torch.utils.data.DataLoader:
    """
    Prepare data loader for training

    Args:
        audio_path: Path to MP3 file
        annotation_csv: Path to timing annotations
        batch_size: Training batch size

    Returns:
        DataLoader instance
    """
    processor = JapaneseSyllableProcessor()
    dataset = JapaneseSyllableDataset(audio_path, annotation_csv, processor)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_length
    )


def collate_variable_length(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function handling variable-length sequences

    Args:
        batch: List of samples

    Returns:
        Batched tensors with padding
    """
    # Get max length in batch
    max_length = max(s['features'].shape[-1] for s in batch)

    # Pad features to max length
    features = [torch.nn.functional.pad(
        s['features'],
        (0, max_length - s['features'].shape[-1])
    ) for s in batch]

    return {
        'features': torch.stack(features),
        'syllable_idx': torch.stack([s['syllable_idx'] for s in batch]),
        'start_times': torch.tensor([s['start_time'] for s in batch]),
        'end_times': torch.tensor([s['end_time'] for s in batch])
    }


# Example usage
def main():
    audio_path = "path/to/your/audio.mp3"
    annotation_csv = "path/to/your/annotations.csv"

    # Create data loader
    train_loader = prepare_training_data(audio_path, annotation_csv)

    # Example of accessing data
    for batch in train_loader:
        features = batch['features']  # Shape: [batch_size, n_mels, time]
        syllables = batch['syllable_idx']  # Shape: [batch_size]

        # Your training loop here
        break


if __name__ == "__main__":
    main()

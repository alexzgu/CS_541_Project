import torch
import torchaudio
import transformers
import numpy as np
from typing import Dict, List
from models.paper_based_approach.implementation import TextlessPhoneAligner


class PhoneAlignmentPreprocessor:
    def __init__(self,
                 sample_rate: int = 16000,
                 max_duration: float = 10.0,
                 phone_inventory: List[str] = None):
        """
        Preprocessing pipeline for phone alignment

        Args:
            sample_rate: Target audio sample rate
            max_duration: Maximum audio clip duration
            phone_inventory: Custom phone set if needed
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Audio preprocessing
        self.audio_transform = torch.nn.Sequential(
            torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )
        )

        # Phoneme preprocessing
        self.g2p_converter = self._load_g2p_converter()

        # Phone inventory (can be expanded)
        self.phone_inventory = phone_inventory or [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
            'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
            'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
        ]

        # Phone to index mapping
        self.phone_to_index = {phone: idx for idx, phone in enumerate(self.phone_inventory)}

    def _load_g2p_converter(self):
        """
        Load grapheme-to-phoneme converter
        Can use libraries like g2p_en or custom implementations
        """
        try:
            import g2p_en
            return g2p_en.G2p()
        except ImportError:
            print("Using fallback G2P method")
            return self._fallback_g2p

    def _fallback_g2p(self, text):
        """
        Fallback grapheme-to-phoneme conversion
        Implement a basic rule-based or dictionary-based conversion
        """
        # Implement basic G2P conversion logic
        pass

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed audio tensor
        """
        # Load audio
        waveform, orig_sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if orig_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate
            )

        # Trim or pad to max duration
        max_length = int(self.max_duration * self.sample_rate)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            padding = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Apply spectral augmentation
        waveform = self._apply_spectral_augmentation(waveform)

        return waveform

    def _apply_spectral_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply time and feature masking

        Args:
            waveform: Input audio tensor

        Returns:
            Augmented audio tensor
        """
        # Time masking
        mask_time_prob = np.random.uniform(0.05, 0.2)
        mask_time_length = int(waveform.shape[1] * mask_time_prob)
        mask_start = np.random.randint(0, waveform.shape[1] - mask_time_length)

        waveform[:, mask_start:mask_start + mask_time_length] = 0

        return waveform

    def convert_text_to_phones(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence

        Args:
            text: Input text

        Returns:
            List of phonemes
        """
        # Convert text to phonemes
        phones = self.g2p_converter(text)

        # Filter out unknown phones
        phones = [p for p in phones if p in self.phone_inventory]

        return phones

    def encode_phones(self, phones: List[str]) -> torch.Tensor:
        """
        Convert phone sequence to tensor

        Args:
            phones: List of phonemes

        Returns:
            Encoded phone tensor
        """
        # Convert phones to indices
        phone_indices = [self.phone_to_index.get(p, -1) for p in phones]

        # Remove unknown phones
        phone_indices = [p for p in phone_indices if p != -1]

        return torch.tensor(phone_indices, dtype=torch.long)


class PhoneAlignmentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_name: str = 'librispeech_clean',
                 split: str = 'train.100',
                 preprocessor: PhoneAlignmentPreprocessor = None):
        """
        Custom dataset for phone alignment

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split
            preprocessor: Preprocessing pipeline
        """
        self.preprocessor = preprocessor or PhoneAlignmentPreprocessor()

        # Load dataset
        self.dataset = datasets.load_dataset(
            dataset_name,
            split=split
        )

        # Optional: Filter and preprocess
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        """
        Additional dataset preprocessing
        """
        # Filter out very long or very short samples
        self.dataset = self.dataset.filter(
            lambda x: (len(x['audio']['array']) / 16000 > 1) and
                      (len(x['audio']['array']) / 16000 < 10)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Prepare item for model training

        Returns:
            Dict with preprocessed audio, phones, and other metadata
        """
        item = self.dataset[idx]

        # Preprocess audio
        audio = self.preprocessor.preprocess_audio(
            item['audio']['path']
        )

        # Convert text to phones
        phones = self.preprocessor.convert_text_to_phones(
            item['text']
        )
        encoded_phones = self.preprocessor.encode_phones(phones)

        return {
            'audio': audio,
            'phones': encoded_phones,
            'text': item['text']
        }


class CurriculumTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        """
        Curriculum learning trainer

        Args:
            model: Phone alignment model
            optimizer: Training optimizer
            device: Computing device
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

    def train_curriculum(self, datasets):
        """
        Curriculum learning training

        Args:
            datasets: List of datasets with increasing complexity
        """
        for iteration, dataset in enumerate(datasets, 1):
            print(f"Curriculum Iteration {iteration}")

            # Create data loader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                num_workers=4
            )

            # Train on current dataset
            self._train_epoch(dataloader, iteration)

            # Optional: Save intermediate model
            self._save_checkpoint(iteration)

    def _train_epoch(self, dataloader, iteration):
        """
        Train for one epoch

        Args:
            dataloader: Training data loader
            iteration: Current curriculum iteration
        """
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            # Move to device
            audio = batch['audio'].to(self.device)
            phones = batch['phones'].to(self.device)

            # Zero grad
            self.optimizer.zero_grad()

            # Forward pass
            loss, alignment = self.model(audio, phones)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()

        # Learning rate scheduling
        self.scheduler.step()

    def _save_checkpoint(self, iteration):
        """
        Save model checkpoint

        Args:
            iteration: Current curriculum iteration
        """
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'phone_alignment_iter_{iteration}.pth')


# Example usage
def main():
    # Preprocessing
    preprocessor = PhoneAlignmentPreprocessor()

    # Datasets with increasing complexity
    datasets = [
        PhoneAlignmentDataset('common_voice', 'train.small'),
        PhoneAlignmentDataset('librispeech_clean', 'train.100'),
        PhoneAlignmentDataset('librispeech_clean', 'train.360')
    ]

    # Model initialization
    model = TextlessPhoneAligner()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Curriculum trainer
    trainer = CurriculumTrainer(model, optimizer)

    # Start curriculum training
    trainer.train_curriculum(datasets)


if __name__ == '__main__':
    main()

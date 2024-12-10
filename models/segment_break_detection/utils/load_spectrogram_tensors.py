import torch
import os
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing preprocessed .pt files
        """
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data_dict = torch.load(file_path, weights_only=True)
        return (
            data_dict['file_idx'],
            data_dict['spectrogram'],
            data_dict['labels']
        )


# Usage example:
if __name__ == "__main__":
    # Load the dataset
    DATA_DIR = "../labeled_spectrogram_tensors"
    dataset = PreprocessedDataset(data_dir=DATA_DIR)
    print(f"Loaded {len(dataset)} samples")

    # Example: access first item
    file_idx, mel_spec, labels = dataset[0]
    print(f"File Index: {file_idx}")
    print(f"Spectrogram shape: {mel_spec.shape}")
    print(f"Labels shape: {labels.shape}")

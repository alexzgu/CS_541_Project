from models.segment_break_detection.utils.dataset import BreakDataset
import torch
import os

# Define paths
#shared_dir = "../../../data/clean"
#AUDIO_DIR = f"{shared_dir}/audio/vocals"
#LABEL_DIR = f"{shared_dir}/segment_breaks"

shared_dir = '../../../data_processing/separate_scripts/'
AUDIO_DIR = f"{shared_dir}/golden_audio"
LABEL_DIR = f"{shared_dir}/golden_breaks"

SAVE_DIR = "../labeled_spectrogram_tensors"
interval_width = 20  # ms

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Initializing dataset with audio_dir={AUDIO_DIR}, label_dir={LABEL_DIR}, interval_width={interval_width}")

# Create the dataset
dataset = BreakDataset(audio_dir=AUDIO_DIR, label_dir=LABEL_DIR, interval_width=interval_width)
# raise(Exception("Stop here"))
print(f"Initialized dataset.")

# Save each spectrogram and labels
for idx, mel_spec, labels in dataset:
    print(f"Processing sample {idx, mel_spec.shape, labels.shape}")
    # Create a dictionary with both tensors
    data_dict = {
        'file_idx': idx,
        'spectrogram': mel_spec,
        'labels': labels
    }

    # Save using torch.save
    save_path = os.path.join(SAVE_DIR, f'{idx}.pt')
    torch.save(data_dict, save_path)
    print(f"Saved sample {idx} to {save_path}")

print(f"Processed and saved {len(dataset)} samples")
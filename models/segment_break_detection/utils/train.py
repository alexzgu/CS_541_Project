import matplotlib.pyplot as plt

from models.segment_break_detection.utils.dataset_old import BreakDataset
from models.segment_break_detection.utils.cnn_rnn import CNNRNN

DATA_DIR = '../../../data'
AUDIO_DIR = f'{DATA_DIR}/clean/audio/vocals'
LABEL_DIR = f'{DATA_DIR}/clean/segment_breaks'

INTERVAL_WIDTH = 20  # ms
NUM_CLASSES = 2  # 0: no break, 1: break

# Initialize the dataset
dataset = BreakDataset(audio_dir=AUDIO_DIR, label_dir=LABEL_DIR)

# Get the first example
file_idx, mel_spec, labels = dataset[0]

# Get the middle M segments
M = 200
total_segments = mel_spec.shape[1]
start_idx = (total_segments - M) // 2
middle_segments = mel_spec[:, start_idx:start_idx + M]
middle_labels = labels[start_idx:start_idx + M]

# Create the plot
plt.figure(figsize=(15, 5))
plt.imshow(middle_segments.numpy(),
           aspect='auto',
           origin='lower',
           cmap='viridis')

# Add vertical lines for break points
break_positions = (middle_labels == True).nonzero()[0]
for pos in break_positions:
    plt.axvline(x=pos, color='red', linestyle='--', alpha=0.7)

plt.colorbar(label='Normalized Log Mel Energy')
plt.xlabel('Time Segments')
plt.ylabel('Mel Frequency Bins')
plt.title(f'Middle 100 Segments of Mel Spectrogram with Break Points (File #{file_idx})')
plt.show()
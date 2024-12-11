import matplotlib.pyplot as plt

from models.segment_break_detection.utils.dataset import BreakDataset
from models.segment_break_detection.utils.cnn_rnn import CNNRNN

DATA_DIR = '../../../data'
AUDIO_DIR = f'{DATA_DIR}/clean/audio/vocals'
LABEL_DIR = f'{DATA_DIR}/clean/segment_breaks'

INTERVAL_WIDTH = 20  # ms
NUM_CLASSES = 2  # 0: no break, 1: break

# Initialize the dataset
dataset = BreakDataset(audio_dir=AUDIO_DIR, label_dir=LABEL_DIR, interval_width=INTERVAL_WIDTH, include_index=False)

# each sample consists of a mel spectrogram and corresponding labels
# mel spectrogram size: [num_intervals, num_mel_bins], labels size: [num_intervals]

# Initialize the model
model = CNNRNN()


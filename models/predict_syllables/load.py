import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchaudio
import torch
import os
from .syllables import one_hot_encoding
from ..wave2vec2 import wave
from .helper import progress_count, calculate_sample_number
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


current_directory = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{current_directory}/../../data'
vocal_dir = f'{data_dir}/clean/audio/vocals'
vocal_index_file = f'{data_dir}/clean/syllables/segment_index.csv'
syllable_dir = f'{data_dir}/clean/syllables'
clip_dir = f'{syllable_dir}/clips'
clip_index_file = f'{syllable_dir}/segment_index.csv'
song_tensor_dir = f'{current_directory}/../tensors/songs'


def get_lstm_dataloader(batch_size, left=0, right=80):
    dataset = get_dataset(left, right)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def convert_songs_to_tensors():
    file_names = os.listdir(vocal_dir)
    mp3_files = filter(lambda x: x.endswith(".mp3"), file_names)
    song_ids = sorted(map(lambda x: int(x[:-4]), mp3_files))

    num_songs = len(song_ids)
    for song_id in song_ids[0:]:
        waveform, sampling_rate = torchaudio.load(f'{vocal_dir}/{song_id}.mp3')
        song_tensor = wave.to_tensors(waveform, sampling_rate, segment_length_ms=10)
        torch.save(song_tensor, f'{current_directory}/../tensors/songs/{song_id}.pt')

        progress_count(song_id, num_songs - 1)


def collate_fn(batch):
    # Batch is a list of tuples (sequence, label)
    sequences, labels = zip(*batch)

    # Sort sequences by length in descending order
    lengths = torch.tensor([len(seq) for seq in sequences])
    sorted_indices = torch.argsort(lengths, descending=True)
    sequences = [sequences[i] for i in sorted_indices]
    labels = [labels[i.item()] for i in sorted_indices]
    labels = torch.from_numpy(np.array(labels)).float()

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)

    return padded_sequences, labels, lengths[sorted_indices]


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        Args:
            sequences (list of torch.Tensor): List of tensors where each tensor is a sequence of shape (seq_len, input_size).
            labels (list of int): List of integer labels corresponding to each sequence.
        """
        assert len(sequences) == len(labels), "Sequences and labels must have the same length"
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def get_dataset(left=0, right=80):
    df = pd.read_csv(vocal_index_file)
    index = {key: group.to_numpy() for key, group in df.groupby('file')}

    sequences = []
    labels = []

    file_names = os.listdir(song_tensor_dir)
    tensor_files = filter(lambda x: x.endswith(".pt"), file_names)
    tensor_ids = sorted(map(lambda x: int(x[:-3]), tensor_files))[left:right]
    num_tensors = len(tensor_ids)

    for num, tensor_id in enumerate(tensor_ids):
        tensor = torch.load(f'{song_tensor_dir}/{tensor_id}.pt')

        if index.get(tensor_id) is None:
            continue

        for syllable_info in index.get(tensor_id):
            start = int(syllable_info[2] / 10)
            end = int(syllable_info[3] / 10)
            syllable_class = syllable_info[4]

            if len(tensor[start:end]) == 0:
                continue
            sequences.append(tensor[start:end])
            labels.append(one_hot_encoding(syllable_class))

        progress_count(num + 1, num_tensors)

    return SequenceDataset(sequences, labels)







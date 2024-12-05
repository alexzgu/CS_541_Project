from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torchaudio
import torch
import os
from .syllables import one_hot_encoding

print(os.getcwd())

from ..wave2vec2 import wave

def get_training_set():
    syllables, classes = load_data()
    X = torch.from_numpy(syllables)
    y = torch.from_numpy(classes)

    dataset = TensorDataset(X, y)
    batch_size = 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_data():
    # load in data with dataloader
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = f'{current_directory}/../../data'
    syllable_dir = f'{data_dir}/clean/syllables'
    clip_dir = f'{syllable_dir}/clips'
    clip_index_file = f'{syllable_dir}/segment_index.csv'

    print(os.getcwd())
    index = pd.read_csv(clip_index_file).to_dict('index')

    syllables = []
    classes = []

    for file in os.listdir(clip_dir)[0:4]:
        if file.endswith('.mp3'):
            clip_id = int(file[:-4])
            waveform, sampling_rate = torchaudio.load(f'{clip_dir}/{file}')
            encoding = wave.to_tensors(waveform, sampling_rate, num_vectors=10)

            syllables.append(encoding.flatten())
            classes.append(one_hot_encoding(index.get(clip_id)['token']))

    return np.array(syllables), np.array(classes)


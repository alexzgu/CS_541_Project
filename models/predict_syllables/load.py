from torch.utils.data import TensorDataset
import pandas as pd
import torchaudio
import torch
import os
from .syllables import one_hot_encoding

print(os.getcwd())

from ..wave2vec2 import wave

def get_training_set():
    syllables, classes = load_data_from_tensors()
    X = syllables
    y = classes

    return TensorDataset(X, y)


current_directory = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{current_directory}/../../data'
syllable_dir = f'{data_dir}/clean/syllables'
clip_dir = f'{syllable_dir}/clips'
clip_index_file = f'{syllable_dir}/segment_index.csv'

def load_data_from_tensors():
    syllables = torch.load(f'{current_directory}/../tensors/syllable_tensors.pt')
    classes = torch.load(f'{current_directory}/../tensors/class_tensors.pt')
    return syllables, classes

def load_data_from_mp3():
    index = pd.read_csv(clip_index_file).to_dict('index')
    syllables = []
    classes = []

    num = 0
    for file in os.listdir(clip_dir)[0:32]:
        if file.endswith('.mp3'):
            clip_id = int(file[:-4])
            waveform, sampling_rate = torchaudio.load(f'{clip_dir}/{file}')
            encoding = wave.to_tensors(waveform, sampling_rate, num_vectors=10)

            syllables.append(torch.flatten(encoding))
            classes.append(one_hot_encoding(index.get(clip_id)['token']))

            num += 1
            print(f'{num}')

    syllable_tensors = torch.stack(syllables)
    class_tensors = torch.stack(classes)

    torch.save(syllable_tensors, f'{current_directory}/../tensors/syllable_tensors.pt')
    torch.save(class_tensors, f'{current_directory}/../tensors/class_tensors.pt')

    return torch.tensor(syllables), torch.tensor(classes)


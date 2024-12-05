from torch.utils.data import TensorDataset
import pandas as pd
import torchaudio
import torch
import os
from .syllables import one_hot_encoding
from ..wave2vec2 import wave

def get_training_set():
    syllables, classes = load_data_from_tensors()
    X = syllables
    y = classes

    return TensorDataset(X, y)


def progress_count(song, current, total):
    ending = '\n' if current == total else ''
    print(f'\rSong {song}: {current}/{total}', end=ending)


current_directory = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{current_directory}/../../data'
vocal_dir = f'{data_dir}/clean/audio/vocals'
vocal_index_file = f'{data_dir}/clean/syllables/segment_index.csv'
syllable_dir = f'{data_dir}/clean/syllables'
clip_dir = f'{syllable_dir}/clips'
clip_index_file = f'{syllable_dir}/segment_index.csv'


def load_data_from_tensors():
    syllables = torch.tensor([])
    classes = torch.tensor([])

    for song_id in range(100):
        try:
            syllables = torch.cat((syllables, torch.load(f'{current_directory}/../tensors/syllables/syllable_tensors-{song_id}.pt')), dim=0)
            classes = torch.cat((classes, torch.load(f'{current_directory}/../tensors/syllables/class_tensors-{song_id}.pt')), dim=0)
        except FileNotFoundError:
            continue

    print(syllables.shape)

    return syllables, classes


def _calculate_sample_number(time_ms, sampling_rate):
    return int((time_ms / 1000) * sampling_rate)


def load_data_from_songs():
    df = pd.read_csv(vocal_index_file)
    index = {key: group.to_numpy() for key, group in df.groupby('file')}

    file_names = os.listdir(vocal_dir)
    mp3_files = filter(lambda x: x.endswith(".mp3"), file_names)
    song_ids = sorted(map(lambda x: int(x[:-4]), mp3_files))

    for song_id in song_ids:
        waveform, sampling_rate = torchaudio.load(f'{vocal_dir}/{song_id}.mp3')
        num_syllables = len((index.get(song_id)))

        syllables = []
        classes = []

        for num, syllable_info in enumerate(index.get(song_id)):
            start = _calculate_sample_number(syllable_info[2], sampling_rate)
            end = _calculate_sample_number(syllable_info[3], sampling_rate)
            syllable = waveform[..., start:end]
            syllable_class = syllable_info[4]
            encoding = wave.to_tensors(syllable, sampling_rate, num_vectors=10)

            syllables.append(torch.flatten(encoding))
            classes.append(one_hot_encoding(syllable_class))

            progress_count(song_id, num + 1, num_syllables)

        syllable_tensors = torch.stack(syllables)
        class_tensors = torch.stack(classes)

        torch.save(syllable_tensors, f'{current_directory}/../tensors/syllable_tensors-{song_id}.pt')
        torch.save(class_tensors, f'{current_directory}/../tensors/class_tensors-{song_id}.pt')

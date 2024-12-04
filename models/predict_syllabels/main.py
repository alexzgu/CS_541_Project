import pandas as pd
import torchaudio
import os

# load in data with dataloader
data_dir = '../../data'
syllable_dir = f'{data_dir}/clean/syllables'
clip_dir = f'{syllable_dir}/clips'
clip_index_file = f'{syllable_dir}/segment_index.csv'

index = pd.read_csv(clip_index_file).to_dict('index')
syllable_clips = dict()

for file in os.listdir(clip_dir):
    if file.endswith('.mp3'):
        clip_id = int(file[:-4])
        syllable_clips[clip_id] = torchaudio.load(f'{clip_dir}/{file}')





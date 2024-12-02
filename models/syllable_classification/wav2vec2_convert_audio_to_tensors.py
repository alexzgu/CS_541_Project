from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch
import os

# load in data with dataloader
data_dir = 'data'
syllable_dir = f'{data_dir}/clean/syllables'
audio_dir = f'{data_dir}/clean/audio/vocals'
# f'{syllable_dir}/clips'  # this contains indexed .mp3 files, each of which is a song
clip_index_file = f'{syllable_dir}/segment_index.csv'
# the clip index file contains the following columns:
# index: the index of the corresponding audio clip
# token: the token (i.e., label) corresponding to the clip (string)

# Initialize wav2vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")


def process_audio(file_path):
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if needed (wav2vec2 expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    # Process through wav2vec2
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state


# Process all files in directory
tensors = {}
for file in os.listdir(audio_dir):
    if file.endswith('.mp3'):
        file_path = os.path.join(audio_dir, file)
        tensors[file] = process_audio(file_path)
        # store the dictionary in the following path: f'{syllable_dir}/tensors.pt'
        print("aaa")
        torch.save(tensors, f'{syllable_dir}/wav2vec2_embeddings.pt')

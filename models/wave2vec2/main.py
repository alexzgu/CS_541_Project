import audio
import os

# load in data with dataloader
data_dir = '../../data'
syllable_dir = f'{data_dir}/clean/syllables'
audio_dir = f'{data_dir}/clean/audio/vocals'
clip_index_file = f'{syllable_dir}/segment_index.csv'


# Example usage
audio_path = "../../data/clean/audio/vocals/0.mp3"
feature_vectors = audio.to_tensors(audio_path, 40)
print(f"Shape of Wav2Vec2 feature vectors: {feature_vectors.shape}")

feature_vectors = audio.to_tensors(audio_path, num_vectors=10)
print(f"Shape of Wav2Vec2 feature vectors: {feature_vectors.shape}")



# Process all files in directory
# tensors = {}
# for file in os.listdir(audio_dir):
#     if file.endswith('.mp3'):
#         file_path = os.path.join(audio_dir, file)
#         tensors[file] = encode_audio_to_wav2vec_vectors(file_path, 10)
#         # store the dictionary in the following path: f'{syllable_dir}/tensors.pt'
#         torch.save(tensors, f'{syllable_dir}/wav2vec2_embeddings.pt')
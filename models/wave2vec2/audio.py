import torchaudio
import wave

def to_tensors(audio_path, segment_length_ms=None, num_vectors=None):
    """
    Encodes an audio file into Wav2Vec2 feature vectors.

    Args:
        audio_path (str): Path to the audio file.
        segment_length_ms (int): Desired length of each encoding segment in milliseconds.
        num_vectors (int): Number of vectors to encode the audio file into.

    Returns:
        numpy.ndarray: Array of Wav2Vec2 feature vectors.
    """

    # Load audio file using torchaudio
    waveform, sampling_rate = torchaudio.load(audio_path)

    return wave.to_tensors(waveform, sampling_rate, segment_length_ms, num_vectors)

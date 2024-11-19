import os

from spleeter.separator import Separator


def split_raw_audio(raw_audio_data_path: str, vocal_audio_data_path: str) -> int:
    """
    Splits the raw audio data into vocal and instrumental audio data, then stores both components.
    Args:
        raw_audio_data_path: Where the raw audio data is stored.
        vocal_audio_data_path: Where the vocal and instrumental audio data will be stored.
            Any existing data with the same file name will be overwritten.

    Returns: 0 if successful, -1 otherwise.
    """

    try:
        os.makedirs(vocal_audio_data_path, exist_ok=True)

        separator = Separator('spleeter:2stems')  # 2-stem for vocal/instrumental split

        # raw_audio_data_path is a directory full of .mp3 files
        for file in os.listdir(raw_audio_data_path):
            if file.endswith('.mp3'):
                separator.separate_to_file(f'{raw_audio_data_path}/{file}', vocal_audio_data_path)
        return 0
    except:
        return -1

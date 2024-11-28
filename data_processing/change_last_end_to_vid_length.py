import pandas as pd
from mutagen.mp3 import MP3
import os


def change_end_for_directory(subtitles_dir: str, audio_dir: str) -> None:
    """
    Change the last entry of 'end' in each csv in the subtitles_dir to the length
    of the corresponding audio file in seconds.
    Args:
        subtitles_dir:
        audio_dir:

    Returns:

    """
    for file in os.listdir(subtitles_dir):
        if file.endswith(".csv"):
            subtitles_file = os.path.join(subtitles_dir, file)
            audio_file = os.path.join(audio_dir, f"{file[:-4]}.mp3")
            change_end_for_file(subtitles_file, audio_file)


def change_end_for_file(subtitles_file: str, audio_file: str) -> None:
    """
    Change the last entry of 'end' in the input subtitles file to the length of the audio file in seconds.
    Args:
        subtitles_file:
        audio_file:

    Returns:

    """
    df = pd.read_csv(subtitles_file)
    vid_length = audio_file_length(audio_file)
    # change last entry of 'end' to vid_length
    df.loc[df.index[-1], 'end'] = vid_length
    df.to_csv(subtitles_file, index=False)

def audio_file_length(audio_file: str) -> int:
    """
    Returns the length of the audio file in seconds.
    Args:
        audio_file: The path to the audio file.

    Returns: The length of the audio file in seconds.
    """
    audio = MP3(audio_file)
    return round(audio.info.length, 3)

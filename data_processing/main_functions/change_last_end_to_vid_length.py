import pandas as pd
from mutagen.mp3 import MP3
import os
from typing import List, Tuple


def change_end_for_directory(subtitles_dir: str, audio_dir: str, segment_index_file: str) -> None:
    """
    Change the last entry of 'end' in each csv in the subtitles_dir to the length
    of the corresponding audio file in seconds.
    Args:
        subtitles_dir:
        audio_dir:
        segment_index_file:

    Returns:
    """

    # the below filters for files with 'end' == inf, then converts the matching rows to a list of 2-tuples
    segment_index_df = pd.read_csv(segment_index_file)
    inf_indices = segment_index_df[segment_index_df['end'] == float('inf')][['index', 'file']]
    inf_indices = list(zip(inf_indices['index'].tolist(), inf_indices['file'].tolist()))

    for file in os.listdir(subtitles_dir):
        if file.endswith(".csv"):
            subtitles_file = os.path.join(subtitles_dir, file)
            audio_file = os.path.join(audio_dir, f"{file[:-4]}.mp3")
            segment_index_df, inf_indices = change_end_for_file(subtitles_file, audio_file, segment_index_df, inf_indices)

    segment_index_df['start'] = segment_index_df['start'].astype(int)
    segment_index_df['end'] = segment_index_df['end'].astype(int)
    segment_index_df.to_csv(segment_index_file, index=False)


def change_end_for_file(subtitles_file: str, audio_file: str, segment_index_df: pd.DataFrame, inf_indices: List[Tuple[int, int]]) -> (pd.DataFrame, List[Tuple[int, int]]):
    """
    Change the last entry of 'end' in the input subtitles file to the length of the audio file in milliseconds.
    Args:
        subtitles_file:
        audio_file:
        segment_index_df:
        inf_indices:
    Returns:
    """
    try:
        MILLISECONDS_IN_SECOND = 1000
        df = pd.read_csv(subtitles_file)
        vid_length = audio_file_length(audio_file)
        # change last entry of 'end' to vid_length
        df.loc[df.index[-1], 'end'] = vid_length

        song_idx = subtitles_file.split('/')[-1].split('.')[0]  # e.g., '1' from '1.csv'
        last_idx = df.index[-1]

        # check whether any of the tuples in inf_indices have a second element with the same value as song_idx
        # if so, pop that tuple from inf_indices and set the entry indexed by the first element of the tuple to vid_length
        for i, tup in enumerate(inf_indices):
            if tup[1] == int(song_idx):
                inf_indices.pop(i)
                segment_index_df.loc[segment_index_df['index'] == tup[0], 'end'] = vid_length * MILLISECONDS_IN_SECOND

        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)

        df.to_csv(subtitles_file, index=False)

    except ValueError:
        print(f"Error changing end for file: {subtitles_file}")
        # print any NA or inf in the dataframe
        print(df[df.isna().any(axis=1)])

    return segment_index_df, inf_indices


def audio_file_length(audio_file: str) -> int:
    """
    Returns the length of the audio file in seconds.
    Args:
        audio_file: The path to the audio file.

    Returns: The length of the audio file in seconds.
    """
    audio = MP3(audio_file)
    return round(audio.info.length, 3)

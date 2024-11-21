from typing import List
import os

import pandas as pd

from data_processing.subtitles.utils.time_ranges import TimeRange, read_ignore_times, compute_silence_ranges
from data_processing.subtitles.utils.row_filtering import remove_hemisphere, compute_overlaps
from data_processing.subtitles.utils.tokens import get_tokens


def clean_subtitles(raw_subtitle_dir: str, ignore_times_dir: str, clean_subtitle_dir: str):
    """
    Args:
        raw_subtitle_dir: Directory containing raw subtitle data.
        ignore_times_dir: Directory containing time ranges to ignore.
        clean_subtitle_dir: Directory to store the cleaned subtitle data.
    """

    # iterates through files in the raw_subtitle_data_path
    for file in os.listdir(raw_subtitle_dir):
        if file.endswith(".csv"):
            ignore_times_file = os.path.join(ignore_times_dir, f"{file[:-4]}.txt")
            raw_subtitles_file = os.path.join(raw_subtitle_dir, file)
            cleaned_subtitles_file = os.path.join(clean_subtitle_dir, file)

            time_ranges_to_ignore = read_ignore_times(ignore_times_file)
            raw_subtitles = pd.read_csv(raw_subtitles_file)
            cleaned_subtitles = clean_subtitles_file(raw_subtitles, time_ranges_to_ignore)
            cleaned_subtitles.to_csv(cleaned_subtitles_file, index=False)

            if cleaned_subtitles.empty:
                print(f"Error cleaning subtitle file: {file}")


def clean_subtitles_file(df: pd.DataFrame, ignore_times: List[TimeRange]) -> pd.DataFrame:
    """
    Args:
        df: DataFrame containing raw subtitle data. It has columns: 'start', 'end', 'line', 'unformatted', 'hiragana'.
        ignore_times: Data telling which time ranges to ignore.

    Returns: Cleaned DataFrame.
    """
    try:
        df = df.drop(columns=['unformatted'])
        df['cleaned_token'] = df['token'].apply(lambda x: ''.join([char for char in x if '\u3040' <= char <= '\u309F']))
        silence_ranges = compute_silence_ranges(df)

        df['overlap'] = compute_overlaps(df)
        df = remove_hemisphere(df)
        df = df.drop(columns=['line'])  # TODO: uncomment this

        df['overlap'] = compute_overlaps(df)
        df = df[~df['overlap']]
        df = insert_silence_and_excluded(df, ignore_times, silence_ranges)

        # 6. with the set of time ranges, insert rows with token = <silence> where there are gaps in the time ranges
        # 7. set the token of a row to <excluded> if any of the below conditions are met:
        #   - the row has 'overlap' = True
        #   - the row has 'ignore' = True
        #   - the row has token = A, where A is not in the set of hiragana characters
        # done

        return df

    except Exception as e:
        print(f"Error in clean_subtitles: {str(e)}")
        return pd.DataFrame()


def insert_silence_and_excluded(df: pd.DataFrame, ig_times: List[TimeRange], sil_times: List[TimeRange]) -> pd.DataFrame:
    """
    Args:
        df: DataFrame containing subtitle data.
        ig_times: List of TimeRange objects representing time ranges to ignore.
        sil_times: List of TimeRange objects representing silent periods.

    Returns: DataFrame with rows inserted for silence and excluded segments.
    """

    return df  # TODO: IMPLEMENT ME



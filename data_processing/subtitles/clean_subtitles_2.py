from typing import List
import os

import pandas as pd

from data_processing.subtitles.utils.character_filtering import to_kana
from data_processing.subtitles.utils.time_ranges import TimeRange, read_time_range_data


def clean_subtitles_2(raw_subtitle_dir: str, time_range_dir: str, clean_subtitle_dir: str):
    """
    Args:
        raw_subtitle_dir: Directory containing raw subtitle data.
        time_range_dir: Directory containing time range data.
        clean_subtitle_dir: Directory to store the cleaned subtitle data.
    """

    # iterates through files in the raw_subtitle_data_path
    for file in os.listdir(raw_subtitle_dir):
        if file.endswith(".csv"):
            ignore_times_file = os.path.join(time_range_dir, f"{file[:-4]}/ignore.txt")
            silence_times_file = os.path.join(time_range_dir, f"{file[:-4]}/silence.txt")
            raw_subtitles_file = os.path.join(raw_subtitle_dir, file)
            cleaned_subtitles_file = os.path.join(clean_subtitle_dir, file)

            time_ranges_to_ignore = read_time_range_data(ignore_times_file)
            silence_time_ranges = read_time_range_data(silence_times_file)
            raw_subtitles = pd.read_csv(raw_subtitles_file)
            cleaned_subtitles = clean_subtitles_file(raw_subtitles, time_ranges_to_ignore, silence_time_ranges)
            cleaned_subtitles.to_csv(cleaned_subtitles_file, index=False)

            if cleaned_subtitles.empty:
                print(f"Error cleaning subtitle file: {file}")


def clean_subtitles_file(df: pd.DataFrame, ignore_times: List[TimeRange], silence_ranges: List[TimeRange]) -> pd.DataFrame:
    """
    Args:
        df: DataFrame containing raw subtitle data. It has columns: 'start', 'end', 'line', 'unformatted', 'hiragana'.
        ignore_times: Data telling which time ranges to ignore.
        silence_ranges: Data telling which time ranges are silent.

    Returns: Cleaned DataFrame.
    """
    try:

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


def insert_silence_and_excluded(df: pd.DataFrame, i_times: List[TimeRange], s_times: List[TimeRange]) -> pd.DataFrame:
    """
    Args:
        df: DataFrame containing subtitle data.
        i_times: List of TimeRange objects representing time ranges to ignore.
        s_times: List of TimeRange objects representing silent periods.

    Returns: DataFrame with rows inserted for silence and excluded segments.
    """

    return df  # TODO: IMPLEMENT ME



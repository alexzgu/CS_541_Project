from typing import List
import os

import pandas as pd

from data_processing.subtitles.utils.character_filtering import to_kana
from data_processing.subtitles.utils.time_ranges import TimeRange, read_time_range_data, compute_silence_ranges
from data_processing.subtitles.utils.row_filtering import remove_hemisphere, compute_overlaps


def clean_subtitles(raw_subtitle_dir: str, ignore_times_dir: str, clean_subtitle_dir: str,
                    output_intermediates: bool = True, intermediate_dir: str = ""):
    """
    Args:
        raw_subtitle_dir: Directory containing raw subtitle data.
        ignore_times_dir: Directory containing time ranges to ignore.
        clean_subtitle_dir: Directory to store the cleaned subtitle data.
        output_intermediates: Whether to output intermediate data (intermediate time ranges and silence time ranges).
        intermediate_dir: Directory to store intermediate data.
    """

    # iterates through files in the raw_subtitle_data_path
    for file in os.listdir(raw_subtitle_dir):
        if file.endswith(".csv"):
            ignore_times_file = os.path.join(ignore_times_dir, f"{file[:-4]}.txt")
            raw_subtitles_file = os.path.join(raw_subtitle_dir, file)
            cleaned_subtitles_file = os.path.join(clean_subtitle_dir, file)

            time_ranges_to_ignore = read_time_range_data(ignore_times_file)
            raw_subtitles = pd.read_csv(raw_subtitles_file)
            cleaned_subtitles, silence_ranges = clean_subtitles_file(raw_subtitles, time_ranges_to_ignore)
            cleaned_subtitles.to_csv(cleaned_subtitles_file, index=False)

            if cleaned_subtitles.empty:
                print(f"Error cleaning subtitle file: {file}")

            if output_intermediates:
                int_file_dir = os.path.join(intermediate_dir, f"{file[:-4]}/")
                os.makedirs(int_file_dir, exist_ok=True)
                silence_file = os.path.join(int_file_dir, "silence.txt")
                ignore_file = os.path.join(int_file_dir, "ignore.txt")

                with open(silence_file, 'w') as f:
                    for time_range in silence_ranges:
                        f.write(f"{time_range.start}:{time_range.end},")
                with open(ignore_file, 'w') as f:
                    for time_range in time_ranges_to_ignore:
                        f.write(f"{time_range.start}:{time_range.end},")


def clean_subtitles_file(df: pd.DataFrame, ignore_times: List[TimeRange]) -> (pd.DataFrame, List[TimeRange]):
    """
    Args:
        df: DataFrame containing raw subtitle data. It has columns: 'start', 'end', 'line', 'unformatted', 'hiragana'.
        ignore_times: Data telling which time ranges to ignore.

    Returns: Cleaned DataFrame and list of time ranges of silence.
    """
    try:
        df = df.drop(columns=['unformatted'])
        df['cleaned_token'] = df['token'].apply(lambda x: to_kana(x))
        silence_ranges = compute_silence_ranges(df)

        df['overlap'] = compute_overlaps(df)
        df = remove_hemisphere(df)
        df = df.drop(columns=['line'])
        df['cleaned_token'] = df['token'].apply(lambda x: to_kana(x, include_katakana=True))

        df['overlap'] = compute_overlaps(df)
        df = df[~df['overlap']]

        return df, silence_ranges

    except Exception as e:
        print(f"Error in clean_subtitles: {str(e)}")
        return pd.DataFrame()

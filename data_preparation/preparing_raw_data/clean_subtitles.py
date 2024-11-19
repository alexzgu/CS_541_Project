import pandas as pd
from typing import Set, List, Tuple, Dict
from dataclasses import dataclass
import os

@dataclass
class TimeRange:
    start: float
    end: float


def clean_subtitles(raw_subtitle_dir: str, ignore_times_dir: str, clean_subtitle_dir: str) -> int:
    """
    Args:
        raw_subtitle_dir: Directory containing raw subtitle data.
        ignore_times_dir: Directory containing time ranges to ignore.
        clean_subtitle_dir: Directory to store the cleaned subtitle data.

    Returns: 0 if successful, -1 otherwise.
    """

    # iterates through files in the raw_subtitle_data_path
    for file in os.listdir(raw_subtitle_dir):
        if file.endswith(".csv"):
            time_ranges_to_ignore = read_ignore_times(os.path.join(ignore_times_dir, f"{file}.csv"))
            raw_subtitles = pd.read_csv(os.path.join(raw_subtitle_dir, file))
            cleaned_subtitles = clean_subtitles_file(raw_subtitles, time_ranges_to_ignore)
            cleaned_subtitles.to_csv(os.path.join(clean_subtitle_dir, file), index=False)
            return 0  # testing only one file for now

    return -1 # TODO: IMPLEMENT


def read_ignore_times(ignore_times_file_path: str) -> List[TimeRange]:
    """
    Reads the ignore_times data from the ignore_times_file_path.
    Args:
        ignore_times_file_path: Path to file containing time ranges to ignore.

    Returns: List of TimeRange objects.
    """
    return [TimeRange(0, 0)]  # TODO: IMPLEMENT


def clean_subtitles_file(df: pd.DataFrame, ignore_times: List[TimeRange]) -> pd.DataFrame:
    """
    Args:
        df: DataFrame containing raw subtitle data. It has columns: 'start', 'end', 'line', 'unformatted', 'hiragana'.
        ignore_times: Data telling which time ranges to ignore.

    Returns: Cleaned DataFrame.
    """
    try:

        # drop unformatted column
        df = df.drop(columns=['unformatted'])

        # go through entire dataset, while (1) grouping rows by line, where each group has a
        # count of rows that contains any hiragana character, and (2) computing a list of time ranges
        # [start, end], not the same as the ignore_times

        # after going through the entire dataset, do the following:
        # 1. group the aggregated rows with line>=0 by whether line>=50 or not. for rows with line=-1, lump them with the
        #    group with the smaller total count of hiragana characters
        # 2. find which rows have overlapping time ranges.
        # 3. see which of the two groups has the smaller total count of hiragana characters. this group will be called group A.
        # 4. all rows belonging group A and with overlap = True are removed from the dataset

        # 5. recompute which rows have overlapping time ranges
        # 6. with the set of time ranges, insert rows with token = <silence> where there are gaps in the time ranges
        # 7. set the token of a row to <excluded> if any of the below conditions are met:
        #   - the row has overlap = True
        #   - the row has ignore = True
        #   - the row has token = a, where a is not in the set of hiragana characters
        # done

        return df

    except Exception as e:
        print(f"Error in clean_subtitles: {str(e)}")
        return pd.DataFrame()

def get_hiragana_set() -> Set[str]:
    """Return the set of hiragana characters."""
    return {
        'あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
        'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と',
        'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ',
        'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り',
        'る', 'れ', 'ろ', 'わ', 'を', 'ん'
    }
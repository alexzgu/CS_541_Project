from typing import Set, List, Tuple, Dict
from dataclasses import dataclass
import os

import pandas as pd
import unicodedata

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
            time_ranges_to_ignore = read_ignore_times(os.path.join(ignore_times_dir, f"{file[:-4]}.txt"))
            raw_subtitles = pd.read_csv(os.path.join(raw_subtitle_dir, file))
            cleaned_subtitles = clean_subtitles_file(raw_subtitles, time_ranges_to_ignore)
            cleaned_subtitles.to_csv(os.path.join(clean_subtitle_dir, file), index=False)
            if cleaned_subtitles.empty:
                print(f"Error cleaning subtitle file: {file}")
            return 0  # testing only one file for now

    return -1 # TODO: IMPLEMENT


def read_ignore_times(ignore_times_file_path: str) -> List[TimeRange]:
    """
    Reads the ignore_times data from the ignore_times_file_path.
    Args:
        ignore_times_file_path: Path to file containing time ranges to ignore.

    Returns: List of TimeRange objects.
    """

    # read in text file as a single string
    with open(ignore_times_file_path, 'r') as file:
        data = file.read()

    # remove whitespace and tabs
    data = data.replace(' ', '').replace('\t', '')

    # if data is empty, return an empty list
    if len(data) == 0:
        return []

    # data is of the form...
    # "start1:end1,start2:end2,start3:end3"

    # split by comma; if there is no comma, then there is only one time range
    data = data.split(',') if ',' in data else [data]
    time_ranges = []

    # iterate through each time range
    for time_range in data:
        if not time_range:
            continue  # skip empty strings
        start, end = time_range.split(':')
        time_ranges.append(TimeRange(float(start), float(end)))

    return time_ranges


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
        # df = df.drop(columns=['line'])  # TODO: uncomment this
        df['overlap'] = compute_overlaps(df)

        df = insert_silence_and_excluded(df)

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
        'あ', 'い', 'う', 'え', 'お',
        'か', 'き', 'く', 'け', 'こ',
        'さ', 'し', 'す', 'せ', 'そ',
        'た', 'ち', 'つ', 'て', 'と',
        'な', 'に', 'ぬ', 'ね', 'の',
        'は', 'ひ', 'ふ', 'へ', 'ほ',
        'ま', 'み', 'む', 'め', 'も',
        'や', 'ゆ', 'よ',
        'ら', 'り', 'る', 'れ', 'ろ',
        'わ', 'を', 'ん',
        'が', 'ぎ', 'ぐ', 'げ', 'ご',
        'ざ', 'じ', 'ず', 'ぜ', 'ぞ',
        'だ', 'ぢ', 'づ', 'で', 'ど',
        'ば', 'び', 'ぶ', 'べ', 'ぼ',
        'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ',
        'きゃ', 'きゅ', 'きょ',
        'しゃ', 'しゅ', 'しょ',
        'ちゃ', 'ちゅ', 'ちょ',
        'にゃ', 'にゅ', 'にょ',
        'ひゃ', 'ひゅ', 'ひょ',
        'みゃ', 'みゅ', 'みょ',
        'りゃ', 'りゅ', 'りょ',
        'ぎゃ', 'ぎゅ', 'ぎょ',
        'じゃ', 'じゅ', 'じょ',
        'びゃ', 'びゅ', 'びょ',
        'ぴゃ', 'ぴゅ', 'ぴょ',
    }


def insert_silence_and_excluded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: DataFrame containing subtitle data.

    Returns: DataFrame with rows inserted for silence and excluded segments.
    """

    return df  # TODO: IMPLEMENT ME


def remove_hemisphere(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decides which set of rows to remove (separated by line>=50 or line<50),
    where the removed set has fewer hiragana characters.
    Args:
        df: DataFrame containing subtitle data.

    Returns: DataFrame with rows removed.
    """

    df = df.copy()

    A = df[(df['line'] >= 0) & (df['line'] < 50)]
    B = df[df['line'] >= 50]
    C = df[df['line'] == -1]

    # deciding which group (A or B) to lump group C with

    c_belongs_with = None

    if A.shape[0] < B.shape[0]:
        A = pd.concat([A, C])
        c_belongs_with = 'A'
    else:
        B = pd.concat([B, C])
        c_belongs_with = 'B'

    A_count = A['cleaned_token'].apply(len).sum()  # top
    B_count = B['cleaned_token'].apply(len).sum()  # bottom

    # if equal
    if A_count == B_count:
        raise(ValueError("A_count and B_count are equal. I know this is improbable but possible, "
                         "but I decided not to deal with this until becomes an actual issue."
                         "I.e., now it is an issue. Please contact Alex."))

    if A_count < B_count:
        if c_belongs_with == 'A':  # remove A
            df = df[~((df['line'] >= 0) & (df['line'] < 50) & df['overlap'])]
        else:
            df = df[~((df['line'] < 50) & df['overlap'])]
    else: # remove B
        if c_belongs_with == 'B':
            df = df[~(((df['line'] >= 50) | (df['line'] == -1)) & df['overlap'])]
        else:
            df = df[~((df['line'] >= 50) & df['overlap'])]

    return df.reset_index(drop=True)


def compute_silence_ranges(df: pd.DataFrame) -> List[TimeRange]:
    """
    Compute ranges of silence from subtitle timing data.

    Args:
        df: DataFrame with 'start' and 'end' columns containing float timestamps

    Returns:
        List of TimeRange objects representing silent periods.
        The last TimeRange will have -1 as its end time.
    """
    # Sort by start time and reset index to ensure proper ordering
    df = df.sort_values(['start', 'end']).reset_index(drop=True)

    # Initialize result list
    silence_ranges = []

    # Check if there's silence at the start
    if len(df) == 0:
        return [TimeRange(0.0, -1)]
    elif df.iloc[0]['start'] > 0:
        silence_ranges.append(TimeRange(0.0, df.iloc[0]['start']))

    # Find gaps between subtitle segments
    for i in range(len(df) - 1):
        current_end = df.iloc[i]['end']
        next_start = df.iloc[i + 1]['start']

        if next_start > current_end:
            silence_ranges.append(TimeRange(current_end, next_start))

    # Add final silence range if there is one
    if len(df) > 0:
        silence_ranges.append(TimeRange(df.iloc[-1]['end'], -1))

    return silence_ranges


def compute_overlaps(df) -> pd.Series:
    """
    Compute overlapping subtitle segments in a DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'start' and 'end' columns containing subtitle timings

    Returns:
    pandas.Series: Boolean series indicating whether each row overlaps with any other row
    """
    n = len(df)
    is_overlap = [False] * n

    # For each row i, look ahead to find any overlaps
    for i in range(n):
        current_end = df.iloc[i]['end']
        # Look at subsequent rows until we find one that starts after current ends
        j = i + 1
        while j < n and df.iloc[j]['start'] < current_end:
            # If we found an overlap, mark both rows
            is_overlap[i] = True
            is_overlap[j] = True
            j += 1

    return pd.Series(is_overlap)

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

        # drop unformatted column
        df = df.drop(columns=['unformatted'])

        df = df.sort_values(by=['start', 'end'])

        df = augment_characters(df)

        # decide whether to remove 'top' or 'bottom' rows

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


def augment_characters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineer the characters (e.g., remove, add, or modify characters within the tokens).

    Args:
        df: DataFrame containing subtitle data.

    Returns: DataFrame with rows removed.
    """

    df = df.copy()

    # def classify_character(char):
    #     if '\u3040' <= char <= '\u309F':
    #         return "Japanese Hiragana"
    #     elif '\u30A0' <= char <= '\u30FF':
    #         return "Japanese Katakana"
    #     elif '\u4E00' <= char <= '\u9FFF':
    #         return "Chinese Character"
    #     elif char.isascii() and char.isalpha():
    #         return "English Alphabet"
    #     elif unicodedata.category(char).startswith('P'):
    #         return "Punctuation"
    #     else:
    #         return "Miscellaneous"
    #

    # remove punctuation, change 々 to repeat the previous character, etc.


    return df


def which_rows_to_remove(df: pd.DataFrame) -> str:
    """
    Args:
        df: DataFrame containing subtitle data.

    Returns: 'top' or 'bottom' depending on which rows to remove.
    """

    df = df.copy()

    df['has_hiragana'] = df['hiragana'].apply(lambda x: any(c in get_hiragana_set() for c in x))

    A = df[df['line'] >= 50]
    B = df[(df['line'] >= 0) & (df['line'] < 50)]
    C = df[df['line'] == -1]

    if A.shape[0] < B.shape[0]:
        A = pd.concat([A, C])
    else:
        B = pd.concat([B, C])

    A_count = A['has_hiragana'].sum()  # top
    B_count = B['has_hiragana'].sum()  # bottom

    # if equal
    if A_count == B_count:
        raise(ValueError("A_count and B_count are equal. I know this is improbable but possible, "
                         "but I decided not to deal with this until becomes an actual issue."
                         "I.e., now it is an issue. Please contact Alex."))

    return 'top' if A_count < B_count else 'bottom'



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
    df = df.sort_values('start').reset_index(drop=True)

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


def compute_overlaps(df):
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

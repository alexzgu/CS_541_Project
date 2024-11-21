from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class TimeRange:
    start: float
    end: float


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

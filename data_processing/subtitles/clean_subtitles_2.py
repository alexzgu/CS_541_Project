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

    silence_token = '<silence>'
    ignore_token = '<ignore>'
    gap_token = '<gap>'

    # 1.) with s_times, insert rows of silence according to s_times (if any)
    # note that none of the silence rows will overlap with any of the existing rows and with each other

    for s_time in s_times:
        silence_row = {
            'start': s_time.start,
            'end': s_time.end,
            'overlap': False,
            'token': silence_token,
            'cleaned_token': silence_token,
        }
        df = df.append(silence_row, ignore_index=True)

    # sort the dataframe by 'start', 'end' time
    df = df.sort_values(by=['start', 'end']).reset_index(drop=True)

    df['ignore'] = False

    # 2.) with i_times...
    # a.) for non-silence rows that overlap with any i_time, set 'ignore' to True
    # b.) for silence rows that are completely covered by an i_time, set 'ignore' to True
    # c.) for silence rows that overlap with an i_time but are not completely covered, split the silence row into two
    #   rows, one with 'ignore' = False and one with 'ignore' = True and token = ignore_token, and adjust the start and end times accordingly
    #   note that the split silence row with 'ignore' = True can be before or after the split silence row with 'ignore' = False

    def is_overlapping(row, time_range) -> bool:
        """
        Returns whether the row overlaps with the time range.
        """
        # overlap is false if end of time range <= start of row
        # or start of time range >= end of row
        return not (time_range.end <= row['start'] or time_range.start >= row['end'])

    for i_time in i_times:
        for index, row in df.iterrows():
            if row['token'] != silence_token:
                if is_overlapping(row, i_time):
                    df.at[index, 'ignore'] = True
            else:  # silence row
                if is_overlapping(row, i_time):
                    i_start_before = i_time.start <= row['start']
                    i_end_after = i_time.end >= row['end']
                    if i_start_before and i_end_after:
                        df.at[index, 'ignore'] = True

                    else:  # split the silence row (here, i_start_before != i_end_after)
                        if i_start_before:
                            # split the silence row with 'ignore' = True
                            silence_row_ignore = row.copy()
                            silence_row_ignore['ignore'] = True
                            silence_row_ignore['cleaned_token'] = ignore_token
                            silence_row_ignore['end'] = i_time.start
                            df = df.append(silence_row_ignore, ignore_index=True)
                            # adjust the start and end times of the original silence row
                            df.at[index, 'start'] = i_time.start
                        else:  # i_end_after
                            # split the silence row with 'ignore' = True
                            silence_row_ignore = row.copy()
                            silence_row_ignore['ignore'] = True
                            silence_row_ignore['cleaned_token'] = ignore_token
                            silence_row_ignore['start'] = i_time.end
                            df = df.append(silence_row_ignore, ignore_index=True)
                            # adjust the start and end times of the original silence row
                            df.at[index, 'end'] = i_time.end


    # note: any silence row should have 'overlap' = False, 'cleaned_token' = silence_token, and 'ignore' = False by default,
    # unless further specified by the above conditions

    df = df.sort_values(by=['start', 'end']).reset_index(drop=True)

    # 3.) scan the entire dataset for any remaining gaps and insert rows with token=gap_token and 'gap' = True

    df['gap'] = False

    for i in range(1, len(df)):
        if df.at[i, 'start'] > df.at[i - 1, 'end']:
            gap_row = {
                'start': df.at[i - 1, 'end'],
                'end': df.at[i, 'start'],
                'overlap': False,
                'token': gap_token,
                'cleaned_token': gap_token,
                'ignore': False,
                'gap': True,
            }
            df = df.append(gap_row, ignore_index=True)

    df = df.sort_values(by=['start', 'end']).reset_index(drop=True)

    # 4.) set any rows with 'overlap' = True or 'cleaned_token' is null to 'other' = True
    df['other'] = df['overlap'] | df['cleaned_token'].isnull()

    # 5.) for any rows with ignore, gap, or other, set 'exclude' to true
    df['exclude'] = df['ignore'] | df['gap'] | df['other']

    return df

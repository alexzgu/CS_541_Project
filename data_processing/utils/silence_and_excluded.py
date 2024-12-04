from typing import List

import pandas as pd

from data_processing.utils.time_ranges import TimeRange


def insert_silence_and_excluded(df: pd.DataFrame, i_times: List[TimeRange], s_times: List[TimeRange], debug: bool = False) -> pd.DataFrame:
    """
    Args:
        df: DataFrame containing subtitle data.
        i_times: List of TimeRange objects representing time ranges to ignore.
        s_times: List of TimeRange objects representing silent periods.
        debug: If True, include intermediate columns in the output DataFrame.

    Returns: DataFrame with rows inserted for silence and excluded segments. New column: 'exclude'.
    """

    silence_token = '<silence>'
    gap_token = '<gap>'

    # 1.) with s_times, insert rows of silence according to s_times (if any)
    # note that none of the silence rows will overlap with any of the existing rows and with each other

    # if end attribute of the last s_time is -1, then change it to float('inf')
    if s_times[-1].end == -1:
        s_times[-1].end = float('inf')

    for s_time in s_times:
        ignored_silence_row = {
            'start': s_time.start,
            'end': s_time.end,
            'overlap': False,
            'token': silence_token,
            'cleaned_token': silence_token,
        }

        df = df.append(ignored_silence_row, ignore_index=True)

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
            if not is_overlapping(row, i_time):
                continue
            df.at[index, 'ignore'] = True

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

    if not debug:
        df = df.drop(columns=['ignore', 'gap', 'other'])

    return df

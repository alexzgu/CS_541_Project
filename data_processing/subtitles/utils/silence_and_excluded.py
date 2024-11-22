from typing import List

import pandas as pd

from data_processing.subtitles.utils.time_ranges import TimeRange


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
            if row['token'] != silence_token:
                df.at[index, 'ignore'] = True
            else:  # silence row
                i_start_before = i_time.start <= row['start']
                i_end_after = i_time.end >= row['end']
                if i_start_before and i_end_after:
                    df.at[index, 'cleaned_token'] = ignore_token
                    df.at[index, 'ignore'] = True

                else:  # split the silence row (i_start_before != i_end_after)
                    if i_start_before:
                        # the time range to ignore partially overlaps with the silence row,
                        # where the time range starts before the silence row
                        # here, the silence row will be split into two rows
                        # the ignored silence row will have start = silence row start, end = time range end
                        # the non-ignored silence row will have start = time range end, end = silence row end
                        df.at[index, 'start'] = i_time.end
                        ignored_silence_row = {
                            'start': row['start'],
                            'end': i_time.end,
                            'overlap': False,
                            'token': silence_token,
                            'cleaned_token': ignore_token,
                            'ignore': True,
                        }
                        df = df.append(ignored_silence_row, ignore_index=True)


                    else:  # i_end_after
                        # the time range to ignore partially overlaps with the silence row,
                        # where the time range ends after the silence row
                        # here, the silence row will be split into two rows
                        # the ignored silence row will have start = time range start, end = silence row end
                        # the non-ignored silence row will have start = silence row start, end = time range start
                        df.at[index, 'end'] = i_time.start
                        ignored_silence_row = {
                            'start': i_time.start,
                            'end': row['end'],
                            'overlap': False,
                            'token': silence_token,
                            'cleaned_token': ignore_token,
                            'ignore': True,
                        }
                        df = df.append(ignored_silence_row, ignore_index=True)


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

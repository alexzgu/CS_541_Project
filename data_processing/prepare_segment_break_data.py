import pandas as pd
import os

SAMPLE_LENGTH = 10  # in milliseconds

def find_segments_breaks(cleaned_subtitle_dir: str, segments_dir: str) -> None:
    """
    For each csv in the cleaned_subtitle_dir, output a text file in segments_dir such that
    the file contains a comma-separated list of all start and end times (no duplicate times).
    Args:
        cleaned_subtitle_dir: Where the cleaned subtitle data (input) is stored.
        segments_dir: Where the segments data (output) will be stored.

    Returns:
    """

    # iterate through files in the cleaned_subtitle_dir
    for file in os.listdir(cleaned_subtitle_dir):
        if file.endswith(".csv"):
            cleaned_subtitle_file = os.path.join(cleaned_subtitle_dir, file)
            segments_data_file = os.path.join(segments_dir, f"{file[:-4]}.csv")

            find_segment_breaks_file(cleaned_subtitle_file, segments_data_file)


def find_segment_breaks_file(cleaned_subtitle_file: str, segments_data_file: str, sample_length=SAMPLE_LENGTH) -> None:
    """
    Generates data for segment break detection. This involves splitting the audio file into discretized intervals, each
    interval of size sample_length (in milliseconds).
    Args:
        cleaned_subtitle_file:
        segments_data_file:
        sample_length: Length of discretized intervals.

    Returns:

    """

    df = pd.read_csv(cleaned_subtitle_file)

    # extract list of end times
    ms_per_s = 1000
    df['end'] = df['end'] * ms_per_s  # in milliseconds
    end_times = df['end'].tolist()

    max_interval = int(end_times[-1])
    discretized_df = pd.DataFrame(columns=['start', 'end'])

    discretized_df['start'] = range(max_interval // sample_length)
    discretized_df['start'] = discretized_df['start'] * sample_length

    discretized_df['end'] = range(1, max_interval // sample_length + 1)
    discretized_df['end'] = discretized_df['end'] * sample_length
    discretized_df['break'] = False

    # for each end time in end_times, set the 'break' column to True for the corresponding row in df2
    for end_time in end_times[:-1]:
        discretized_df.loc[(discretized_df['start'] <= end_time) & (discretized_df['end'] > end_time), 'break'] = True

    # write entire discretized_df to segments_data_file
    discretized_df.to_csv(segments_data_file, index=False)

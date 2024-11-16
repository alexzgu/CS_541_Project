def find_segments(cleaned_subtitle_data_path: str, segments_data_path: str) -> int:
    """
    For each csv in the cleaned_subtitle_data_path, output a text file in segments_data_path such that
    the file contains a comma-separated list of all start and end times (no duplicate times).
    Args:
        cleaned_subtitle_data_path: Where the cleaned subtitle data (input) is stored.
        segments_data_path: Where the segments data (output) will be stored.

    Returns: 0 if successful, -1 otherwise.
    """
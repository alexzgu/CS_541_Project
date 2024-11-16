def split_raw_audio(raw_audio_data_path: str, vocal_audio_data_path: str) -> int:
    """
    Splits the raw audio data into vocal and instrumental audio data, then stores the vocal audio data.
    Args:
        raw_audio_data_path: Where the raw audio data is stored.
        vocal_audio_data_path: Where the vocal audio data will be stored.
            Any existing data with the same file name will be overwritten.

    Returns: 0 if successful, -1 otherwise.
    """

    return -1  # TODO: Implement this function


def clean_subtitles(raw_subtitle_data_path: str, ignore_times_file_path: str, clean_subtitle_data_path: str) -> int:
    """
    Cleans the raw subtitle data and stores the cleaned subtitle data.
    Args:
        raw_subtitle_data_path: Where the raw subtitle data is stored.
        ignore_times_file_path: Path to the file containing time ranges to ignore.
        clean_subtitle_data_path: Where the cleaned subtitle data will be stored.
            Any existing data with the same file name will be overwritten.

    Returns: 0 if successful, -1 otherwise.
    """

    """
    The cleaned subtitle data will have additional rows corresponding to <silence> tokens,
    and may have additional rows corresponding to <excluded> tokens.
    """

    return -1  # TODO: Implement this function
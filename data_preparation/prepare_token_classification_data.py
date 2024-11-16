def segment_audio(vocal_audio_data_path: str, clean_subtitle_data_path: str,
                  segmented_audio_data_path: str, segment_index_file_path: str) -> int:
    """
    Segments the vocal audio data into smaller audio clips based on the cleaned subtitle data.
    Each clip is indexed, and information about its file origin, start time, end time, and label (token)
    are stored in a segment index file.
    Args:
        vocal_audio_data_path: Where the vocal audio data is stored.
        clean_subtitle_data_path: Where the cleaned subtitle data is stored.
        segmented_audio_data_path: Where the segmented audio data will be stored.
            Any existing data with the same file name will be overwritten.
        segment_index_file_path:  Where the segment index file will be stored.
            If it already exists, the data will be appended to the existing file.

    Returns: 0 if successful, -1 otherwise.
    """

    return -1  # TODO: Implement this function


def sort_segments(segmented_audio_data_path: str, segment_index_file_path: str, segment_buckets_directory: str) -> int:
    """
    Sorts the segmented audio data based on the segment's label (token),
    and stores the sorted audio data (copied, not moved inplace).
    Args:
        segmented_audio_data_path: Where the segmented audio data is stored.
        segment_index_file_path: Where the segment index file is stored.
        segment_buckets_directory: Where the sorted audio data will be stored.
            All audio clips with the same label will be stored in the same directory,
            which is named after the label.

    Returns: 0 if successful, -1 otherwise.
    """

    return -1  # TODO: Implement this function
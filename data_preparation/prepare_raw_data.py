import pandas as pd
import youtube_dl

def download_raw_audio_data(raw_subtitle_data_path: str, raw_audio_data_path: str, index_file_path: str) -> int:
    """
    Downloads raw audio data from the internet, for the indexed raw subtitle data available.
    Args:
        raw_subtitle_data_path: Where the raw subtitle data is stored.
        raw_audio_data_path: Where the raw audio data will be loaded into.
            Any existing data with the same file name will be overwritten.
        index_file_path: Path to the index TSV file containing Index to ID mapping.

    Returns: 0 if successful, -1 otherwise.
    """

    # these are YouTube videos, and the IDs are the video IDs
    index_df = pd.read_csv(index_file_path, sep='\t')
    index_df = index_df.dropna()
    index_df = index_df.reset_index(drop=True)


    for i in range(len(index_df)):
        video_id = index_df.loc[i, 'ID']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        video_path = f'{raw_audio_data_path}/{video_id}.mp3'

        # download the audio
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': video_path,
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            print(f'Error downloading video {video_id}: {e}')
            return -1

    return 0


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
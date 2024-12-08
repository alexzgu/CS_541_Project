import os
import pandas as pd
from pydub import AudioSegment

def segment_audio(vocal_audio_dir: str, clean_subtitle_dir: str,
                  segmented_audio_dir: str, segment_index_file_path: str, padding: int = 0, produce_audio_files: bool = False) -> int:
    """
    Segments the vocal audio data into smaller audio clips based on the cleaned subtitle data.
    Each clip is indexed, and information about its file origin, start time, end time, and label (token)
    are stored in a segment index file.
    Args:
        produce_audio_files:
        vocal_audio_dir: Where the vocal audio data is stored.
        clean_subtitle_dir: Where the cleaned subtitle data is stored.
        segmented_audio_dir: Where the segmented audio data will be stored.
            Any existing data with the same file name will be overwritten.
        segment_index_file_path:  Where the file containing indices for segments will be stored.
            If it already exists, the data will be appended to the existing file.
        padding: Number of milliseconds to pad the start and end times of each segment.

    Returns: 0 if successful, -1 otherwise.
    """
    segment_index_file = os.path.join(segment_index_file_path)

    # create segmented audio directory if it doesn't exist
    # if not os.path.exists(segmented_audio_dir):
    #     os.makedirs(segmented_audio_dir)

    # write initial header to segment index file
    # Index, File, Start, End, Token
    with open(segment_index_file, 'w') as f:
        f.write("index,file,start,end,token\n")

    # iterates through files in the clean_subtitle_data_path
    for file in os.listdir(clean_subtitle_dir):
        if file.endswith(".csv"):
            print(f"Segmenting audio for file: {file}")
            subtitles_file = os.path.join(clean_subtitle_dir, file)
            vocal_audio_file = os.path.join(vocal_audio_dir, f"{file[:-4]}.mp3")
            segment_audio_file(subtitles_file, vocal_audio_file, segmented_audio_dir,
                               segment_index_file, padding=padding, produce_audio_files=produce_audio_files)
    return 0


def segment_audio_file(subtitles_file: str, vocal_audio_file: str, segmented_audio_dir: str,
                       segment_index_file: str, padding:int=0, produce_audio_files: bool = False):
    """
    Segments the vocal audio data into smaller audio clips based on the cleaned subtitle data.
    Args:
        produce_audio_files:
        subtitles_file:
        vocal_audio_file:
        segmented_audio_dir:
        segment_index_file:
        padding: Number of milliseconds to pad the start and end times of each segment
        (except the first segment's start). Default padding is 0 ms.

    Returns:

    """
    file_idx = subtitles_file.split('/')[-1].split('.')[0]

    df = pd.read_csv(subtitles_file)
    vocal_audio = AudioSegment.from_file(vocal_audio_file)

    # filter for rows where exclude = True
    df = df[df['exclude'] == False].reset_index(drop=True)
    for idx, row in df.iterrows():
        NUM_SEGMENTS_PER_SECOND = 1000
        start = row['start'] * NUM_SEGMENTS_PER_SECOND
        end = row['end']  * NUM_SEGMENTS_PER_SECOND

        token = row['token']
        segmented_audio_file = os.path.join(segmented_audio_dir, f"{file_idx}_{idx}.mp3")
        with open(segment_index_file, 'a') as f:
            f.write(f"{file_idx}_{idx},{segmented_audio_file},{start},{end},{token}\n")

        # Note: padding is applied below, not to the giant index file above
        if produce_audio_files:
            start_padded = start - padding if idx != 0 else start
            end_padded = end + padding if idx != len(df) - 1 else end
            segment = vocal_audio[start_padded:end_padded]
            segment.export(segmented_audio_file, format="mp3")


def reindex_audio_segments(old_index_file: str, new_index_file: str, audio_clip_dir: str,
                           audio_files_present: bool = False) -> None:
    df = pd.read_csv(old_index_file)
    df['old_idx'] = df['index'].copy()
    df['index'] = df.index

    # renames the corresponding audio files
    if audio_files_present:
        df[['old_idx', 'index']].apply(lambda x: rename_audio_clip(x, audio_clip_dir), axis=1)

    df = df.drop(columns=['old_idx'])

    # entries structured as path/.../#_#.csv
    df['file'] = df['file'].apply(lambda x: x.split('/')[-1].split('.')[0].split('_')[0])
    df.to_csv(new_index_file, index=False)


def rename_audio_clip(row, audio_clip_dir: str) -> None:
    old_idx = row[0]
    new_idx = row[1]

    current_audio_clip_file = os.path.join(audio_clip_dir, f"{old_idx}.mp3")
    new_audio_clip_file = os.path.join(audio_clip_dir, f"{new_idx}.mp3")

    os.rename(current_audio_clip_file, new_audio_clip_file)
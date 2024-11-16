from prepare_raw_data import *

if __name__ == '__main__':
    data_dir = '../data'

    raw_audio_data_path = f'{data_dir}/raw/audio_raw'
    raw_subtitle_data_path = f'{data_dir}/raw/subtitles_raw'
    vocal_audio_data_path = f'{data_dir}/song_vocals'
    ignore_times_file_path = f'{data_dir}/ignore_times.txt'
    clean_subtitle_data_path = f'{data_dir}/subtitles_clean'
    index_file_path = f'{data_dir}/index.tsv'

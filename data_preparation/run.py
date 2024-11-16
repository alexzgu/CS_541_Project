from prepare_raw_data import *

if __name__ == '__main__':
    data_dir = '../data'
    raw_audio_data_path = f'{data_dir}/raw_audio'
    vocal_audio_data_path = f'{data_dir}/vocal_audio'
    raw_subtitle_data_path = f'{data_dir}/raw_subtitles'
    ignore_times_file_path = f'{data_dir}/ignore_times.txt'
    clean_subtitle_data_path = f'{data_dir}/clean_subtitles'
    index_file_path = f'{data_dir}/index.tsv'

    if download_raw_audio_data(raw_subtitle_data_path, raw_audio_data_path, index_file_path) != 0:
        print('Error preparing raw audio data')
        exit(-1)
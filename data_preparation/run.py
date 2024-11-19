from data_preparation.preparing_raw_data.split_audio import split_raw_audio
from data_preparation.preparing_raw_data.clean_subtitles import clean_subtitles

if __name__ == '__main__':
    data_dir = '../data'

    raw_audio_data_path = f'{data_dir}/raw/audio_raw'
    raw_subtitle_data_path = f'{data_dir}/raw/subtitles_raw'
    vocal_audio_data_path = f'{data_dir}/song_vocals'
    ignore_times_data_path = f'{data_dir}/clips_to_exclude'
    clean_subtitle_data_path = f'{data_dir}/subtitles_clean'
    index_file_path = f'{data_dir}/index.tsv'


    # print("Splitting raw audio into vocals and instrumentals...")
    # process_1 = split_raw_audio(raw_audio_data_path, vocal_audio_data_path)
    #
    # if process_1 == 0:
    #     print("Splitting audio data successful.")
    # else:
    #     print("Error splitting audio data.")

    print("Cleaning raw subtitle data...")
    process_2 = clean_subtitles(raw_subtitle_data_path, ignore_times_data_path, clean_subtitle_data_path)

    if process_2 == 0:
        print("Cleaning subtitle data successful.")
    else:
        print("Error cleaning subtitle data.")
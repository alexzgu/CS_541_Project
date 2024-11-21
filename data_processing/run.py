from data_processing.subtitles.clean_subtitles import clean_subtitles

if __name__ == '__main__':
    data_dir = '../data'
    clean_subtitles_path = f'{data_dir}/clean/subtitles'
    tokens_file_path = f'{data_dir}/config/tokens.txt'

    raw_subtitles_dir = f'{data_dir}/raw/subtitles'
    raw_subtitles_path = f'{raw_subtitles_dir}/subtitle_files'
    ignore_times_path = f'{raw_subtitles_dir}/clips_to_exclude'
    index_file_path = f'{raw_subtitles_dir}/index.tsv'


    print("Cleaning raw subtitle data...")
    clean_subtitles(raw_subtitles_path, ignore_times_path, clean_subtitles_path)
    print("Cleaned subtitle data.")
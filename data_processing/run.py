from data_processing.clean_subtitles import clean_subtitles

if __name__ == '__main__':
    data_dir = '../data'

    raw_subtitles_dir = f'{data_dir}/raw/subtitles'
    raw_subtitles_path = f'{raw_subtitles_dir}/subtitle_files'
    ignore_times_path = f'{raw_subtitles_dir}/clips_to_exclude'
    index_file_path = f'{raw_subtitles_dir}/index.tsv'

    clean_subtitles_path = f'{data_dir}/clean/subtitles/subtitle_files'


    print("Cleaning raw subtitle data...")
    process_2 = clean_subtitles(raw_subtitles_path, ignore_times_path, clean_subtitles_path)

    if process_2 == 0:
        print("Cleaning subtitle data successful.")
    else:
        print("Error cleaning subtitle data.")
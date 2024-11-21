from data_processing.clean_subtitles import clean_subtitles

if __name__ == '__main__':
    data_dir = '../data'

    raw_dir = f'{data_dir}/raw'
    raw_subtitles_path = f'{raw_dir}/subtitles'
    ignore_times_path = f'{raw_dir}/clips_to_exclude'

    clean_subtitles_path = f'{data_dir}/subtitles_clean'
    index_file_path = f'{data_dir}/index.tsv'

    print("Cleaning raw subtitle data...")
    process_2 = clean_subtitles(raw_subtitles_path, ignore_times_path, clean_subtitles_path)

    if process_2 == 0:
        print("Cleaning subtitle data successful.")
    else:
        print("Error cleaning subtitle data.")
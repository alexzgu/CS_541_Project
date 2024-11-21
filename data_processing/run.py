from data_processing.subtitles.clean_subtitles_1 import clean_subtitles

if __name__ == '__main__':
    data_dir = '../data'
    clean_subtitles_path = f'{data_dir}/clean/subtitles'
    tokens_file_path = f'{data_dir}/config/tokens.txt'

    raw_subtitles_dir = f'{data_dir}/raw/subtitles'
    raw_subtitles_path = f'{raw_subtitles_dir}/subtitle_files'
    ignore_times_path = f'{raw_subtitles_dir}/clips_to_exclude'
    index_file_path = f'{raw_subtitles_dir}/index.tsv'

    stage_1_dir = f'{data_dir}/processed/subtitles/stage_1'

    print("Cleaning raw subtitle data...")
    clean_subtitles(raw_subtitles_path, ignore_times_path, f'{stage_1_dir}/subtitle_files',
                    output_intermediates=True, intermediate_dir=f'{stage_1_dir}/time_ranges')
    print("Cleaned subtitle data.")
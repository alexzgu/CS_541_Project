import warnings

from data_processing.main_functions.clean_subtitles import clean_subtitles
from data_processing.main_functions.change_last_end_to_vid_length import change_end_for_directory
from data_processing.prepare_segment_break_data import find_segments_breaks
from data_processing.prepare_token_classification_data import segment_audio, reindex_audio_segments
from data_processing.utils.reduce_silence import reduce_silence_for_directory

# suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    # paths
    data_dir = '../data'

    clean_subtitles_path = f'{data_dir}/clean/subtitles'
    processed_subtitles_path = f'{data_dir}/processed/subtitles'
    tokens_file_path = f'{data_dir}/config/tokens.txt'

    raw_subtitles_dir = f'{data_dir}/raw/subtitles'
    raw_subtitles_path = f'{raw_subtitles_dir}/subtitle_files'
    ignore_times_path = f'{raw_subtitles_dir}/clips_to_exclude'

    index_file_path = f'{raw_subtitles_dir}/index.tsv'
    tokens_path = f'{data_dir}/config/tokens.txt'

    raw_vocals_dir = f'{data_dir}/raw/audio'
    clean_vocals_dir = f'{data_dir}/clean/audio/vocals'
    syllables_dir = f'{data_dir}/clean/syllables'
    segment_index_file_path = f'{syllables_dir}/segment_index.csv'

    syllable_vocals_dir = f'{syllables_dir}/clips'
    segment_break_dir = f'{data_dir}/clean/segment_breaks'

    # do things; NOTE: never run clean_subtitles again. we got what we needed from it.
    # print("Cleaning raw subtitle data...")
    # clean_subtitles(raw_subtitles_path, ignore_times_path, processed_subtitles_path, tokens_file_path)
    # print("Cleaned subtitle data.")
    #
    # print("Segmenting audio data...")
    # segment_audio(clean_vocals_dir, processed_subtitles_path, syllable_vocals_dir,
    #               segment_index_file_path, padding=50)
    # print("Segmented audio data.")
    #
    # # # must run indexing right after segmenting
    # print("Indexing audio data...")
    # reindex_audio_segments(segment_index_file_path,
    #                        segment_index_file_path,
    #                        syllable_vocals_dir)
    # print("Indexed audio data.")
    #
    # print("Changing last end to vid length...")
    # change_end_for_directory(processed_subtitles_path, raw_vocals_dir, segment_index_file_path)
    # print("Changed last end to vid length.")

    # reduce silence
    # print("Reducing silence...")
    # reduce_silence_for_directory(processed_subtitles_path, clean_vocals_dir,
    #                              clean_subtitles_path)
    # print("Reduced silence.")

    # for segment break detection
    # print("Finding segment breaks...")
    # find_segments_breaks(clean_subtitles_path, segment_break_dir, sample_length=20)
    # print("Found segment breaks.")

    # scuffed thing; do not run this
    sep_script_dir = f'separate_scripts'
    golden_csv_dir = f'{sep_script_dir}/golden_csvs/processed'
    clean_golden_csv_dir = f'{sep_script_dir}/golden_csvs/clean'
    golden_vocals_dir = f'{sep_script_dir}/golden_audio'
    segment_idx_file = f'{sep_script_dir}/segment_index.csv'
    segment_breaks_dir = f'{sep_script_dir}/golden_breaks'

    print("Segmenting audio data...")
    segment_audio(golden_vocals_dir, golden_csv_dir, 'separate_scripts/golden_syllables',
                  segment_idx_file, padding=50)
    print("Segmented audio data.")

    print("Indexing audio data...")
    reindex_audio_segments(segment_idx_file, segment_idx_file,
                           'separate_scripts/golden_syllables')

    print("Finding segment breaks...")
    find_segments_breaks(golden_csv_dir, segment_breaks_dir, sample_length=20)
    print("Found segment breaks.")

    # read in segment break idx file and cast start and end times to int
    import pandas as pd
    df = pd.read_csv(segment_idx_file)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df.to_csv(segment_idx_file, index=False)
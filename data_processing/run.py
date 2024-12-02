import warnings

from data_processing.main_functions.clean_subtitles import clean_subtitles
from data_processing.main_functions.change_last_end_to_vid_length import change_end_for_directory
from data_processing.prepare_segment_break_data import find_segments_breaks
from data_processing.prepare_token_classification_data import segment_audio, reindex_audio_segments

# suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    data_dir = '../data'
    clean_subtitles_path = f'{data_dir}/clean/subtitles'
    tokens_file_path = f'{data_dir}/config/tokens.txt'

    raw_subtitles_dir = f'{data_dir}/raw/subtitles'
    raw_subtitles_path = f'{raw_subtitles_dir}/subtitle_files'
    ignore_times_path = f'{raw_subtitles_dir}/clips_to_exclude'

    index_file_path = f'{raw_subtitles_dir}/index.tsv'
    tokens_path = f'{data_dir}/config/tokens.txt'

    raw_vocals_dir = f'{data_dir}/raw/audio'
    syllables_dir = f'{data_dir}/clean/syllables'
    segment_index_file_path = f'{syllables_dir}/segment_index.csv'

    syllable_vocals_dir = f'{syllables_dir}/clips'
    segment_break_dir = f'{data_dir}/clean/segment_breaks'

    # stage_1_dir = f'{data_dir}/processed/subtitles/stage_1'
    # time_range_dir = f'{stage_1_dir}/time_ranges'

    # print("Cleaning raw subtitle data...")
    # clean_subtitles(raw_subtitles_path, ignore_times_path, clean_subtitles_path, tokens_file_path)
    # print("Cleaned subtitle data.")
    #
    # print("Segmenting audio data...")
    # segment_audio(raw_vocals_dir, clean_subtitles_path, syllable_vocals_dir, segment_index_file_path)
    # print("Segmented audio data.")

    # must run indexing right after segmenting
    # print("Indexing audio data...")
    # reindex_audio_segments(segment_index_file_path,
    #                        segment_index_file_path,
    #                        syllable_vocals_dir)
    # print("Indexed audio data.")

    # print("Changing last end to vid length...")
    # change_end_for_directory(clean_subtitles_path, raw_vocals_dir, segment_index_file_path)
    # print("Changed last end to vid length.")

    # converting all start/end times in segment_index_file from float to int

    print("Finding segment breaks...")
    find_segments_breaks(clean_subtitles_path, segment_break_dir)
    print("Found segment breaks.")
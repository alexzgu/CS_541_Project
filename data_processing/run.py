from data_processing.subtitles.clean_subtitles import clean_subtitles
from data_processing.subtitles.prepare_token_classification_data import segment_audio, reindex_audio_segments
import warnings

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

    raw_vocals_dir = f'{data_dir}/raw/audio'
    syllables_dir = f'{data_dir}/clean/syllables'
    segment_index_file_path = f'{syllables_dir}/segment_index.csv'
    reindexed_segment_index_file_path = f'{syllables_dir}/reindexed_segment_index.csv'

    syllable_vocals_dir = f'{syllables_dir}/clips'

    # stage_1_dir = f'{data_dir}/processed/subtitles/stage_1'
    # time_range_dir = f'{stage_1_dir}/time_ranges'

    # print("Cleaning raw subtitle data...")
    # clean_subtitles(raw_subtitles_path, ignore_times_path, clean_subtitles_path, # f'{stage_1_dir}/subtitle_files',
    #                 output_intermediates=True, intermediate_dir=time_range_dir)
    # print("Cleaned subtitle data.")

    # print("Cleaning raw subtitle data...")
    # clean_subtitles_2(f'{stage_1_dir}/subtitle_files', time_range_dir, clean_subtitles_path)

    # print("Segmenting audio data...")
    # segment_audio(raw_vocals_dir, clean_subtitles_path, syllable_vocals_dir, segment_index_file_path)
    # print("Segmented audio data.")

    print("Indexing audio data...")
    reindex_audio_segments(segment_index_file_path,
                           reindexed_segment_index_file_path,
                           syllable_vocals_dir)
    print("Indexed audio data.")

# TODO: create segment break data

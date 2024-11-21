from split_audio import split_raw_audio

# NOTE: Feel free to delete/change any part of this code.
# I'm just putting this here in case it helps at all.

if __name__ == '__main__':
    data_dir = '../data'

    raw_audio_path = f'{data_dir}/raw/audio'
    clean_audio_path = f'{data_dir}/clean/audio'

    index_file_path = f'{data_dir}/index.tsv'

    print("Splitting raw audio into vocals and instrumentals...")
    process_1 = split_raw_audio(raw_audio_path, clean_audio_path)

    if process_1 == 0:
        print("Splitting audio data successful.")
    else:
        print("Error splitting audio data.")
# prompt: write a function that (for specified input csv, input mp3, output csv directories), do this for all csv's

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reduce_silence_for_directory(input_csv_dir, input_mp3_dir, output_csv_dir):
    """
    Processes all CSV files in the input directory, trims silence from corresponding audio segments,
    and saves the updated CSV files to the output directory.
    """

    for filename in os.listdir(input_csv_dir):
        if filename.endswith(".csv"):
            try:
                # Construct file paths
                csv_filepath = os.path.join(input_csv_dir, filename)
                mp3_filename = filename.replace(".csv", ".mp3")  # Assuming corresponding .mp3 file exists
                mp3_filepath = os.path.join(input_mp3_dir, mp3_filename)
                output_csv_filepath = os.path.join(output_csv_dir, filename)

                # Load audio file
                audio, sr = librosa.load(mp3_filepath)

                # Load dataframe
                df = pd.read_csv(csv_filepath)

                # Process each row (segment) in the DataFrame
                for index, row in df.iterrows():
                    if row['token'] != '<silence>':
                        new_start, new_end, spectrogram = process_audio_segment(row, audio, sr)
                        df.at[index, 'start'] = new_start
                        df.at[index, 'end'] = new_end
                        #Optional: display spectrogram, uncomment the lines below if needed.
                        # if spectrogram is not None:
                        #     plt.figure(figsize=(14, 5))
                        #     librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
                        #     plt.colorbar(format='%+2.0f dB')
                        #     plt.title(f'Spectrogram (Trimmed and Aligned) - Row {index} - File {filename}')
                        #     plt.show()

                # for any rows with token = '<silence>', change their start and end times as needed to
                # match the new start and end times of the rows adjacent to it
                for index, row in df.iterrows():
                    if row['token'] == '<silence>':
                        if index > 0:
                          prev_row = df.iloc[index - 1]
                          df.at[index, 'start'] = prev_row['end']
                        elif  index < len(df) - 1:
                          next_row = df.iloc[index + 1]
                          df.at[index, 'end'] = next_row['start']

                df['start'] = df['start'].round(3)
                df['end'] = df['end'].round(3)

                # iterate through all rows, and if there is a gap between the end of one row and the start of the next row,
                # insert a new row with <silence> token and the start and end times of the gap
                for index, row in df.iterrows():
                    if index < len(df) - 1:
                        next_row = df.iloc[index + 1]
                        gap = next_row['start'] - row['end']
                        if gap > 0:
                            new_row = pd.DataFrame([[row['end'], next_row['start'], '<silence>']], columns=['start', 'end', 'token'])
                            df = pd.concat([df.iloc[:index + 1], new_row, df.iloc[index + 1:]]).reset_index(drop=True)

                # Save the updated dataframe
                df.to_csv(output_csv_filepath, index=False)
                print(f"Processed {filename} and saved to {output_csv_filepath}")

            except FileNotFoundError:
                print(f"Error: Corresponding .mp3 file not found for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def process_audio_segment(row, audio, sr):
    try:
        # Extract the segment of interest
        start_time = row['start']
        end_time = row['end']
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = audio[start_sample:end_sample]

        # Trim silence from the segment
        trimmed_segment = librosa.effects.trim(segment, top_db=20)[0]

        original_length = len(segment)
        trimmed_length = len(trimmed_segment)

        if trimmed_length > 0 and original_length > 0:
            optimal_shift = 0
            new_start_time = start_time + optimal_shift / sr
            new_end_time = new_start_time + (trimmed_length / sr)

            # Generate the spectrogram for the trimmed segment
            X = librosa.stft(trimmed_segment)
            Xdb = librosa.amplitude_to_db(abs(X))
            return new_start_time, new_end_time, Xdb
        else:
            print("Cannot process segment. Either original or trimmed segment is empty.")
            return start_time, end_time, None
    except Exception as e:
        print(f"Error processing segment: {e}")
        return start_time, end_time, None
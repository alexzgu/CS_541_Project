import pandas as pd
from pathlib import Path


def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder % 1) * 1000)
    seconds_int = int(seconds_remainder)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"


def create_srt_from_csv(csv_path, output_path):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Generate SRT content
    srt_content = []
    for index, row in df.iterrows():
        start_time = seconds_to_srt_time(row['start'])
        end_time = seconds_to_srt_time(row['end'])
        token = row['token'] + " (excluded)" if row['exclude'] else row['token']
        srt_entry = f"{index + 1}\n{start_time} --> {end_time}\n{token}\n"
        srt_content.append(srt_entry)

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_content))


def process_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each CSV file in the input directory
    for csv_file in Path(input_dir).glob('*.csv'):
        # Create corresponding SRT filename
        srt_file = Path(output_dir) / f"{csv_file.stem}.srt"

        # Process the file
        create_srt_from_csv(csv_file, srt_file)
        print(f"Processed {csv_file.name} -> {srt_file.name}")

# Example usage:
data_dir = "../../data"
clean_subtitles_dir = f"{data_dir}/clean/subtitles"
srt_output_dir = f"{data_dir}/processed/srts"
process_directory(clean_subtitles_dir, srt_output_dir)

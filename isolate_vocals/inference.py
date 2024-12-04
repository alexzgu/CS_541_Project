import argparse
from audio_separator.separator import Separator
import glob
import os

def main(input_dir, output_dir):
    separator = Separator(output_dir=output_dir)

    separator.load_model()

    mp3_files = glob.glob(os.path.join(input_dir, '*.mp3'))

    for mp3_file in mp3_files:
        print(f"Processing file: {mp3_file}")
        output_files = separator.separate(mp3_file)
        print(f"Output files: {output_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio file separator script.")
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Path to the directory containing input audio files."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Path to the directory where output files will be saved."
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

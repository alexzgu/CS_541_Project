## Neccessary Dependencies:

FFmpeg will be required to run the voice_separator library.

## Install FFmpeg
**Linux**
- bash
sudo apt update
sudo apt install ffmpeg

**Windows**
choco install ffmpeg

# Install required libraries
pip install -r requirement.txt

# Run the script
python inference.py --input_dir ../data/raw/audio --output_dir ../data/clean/audio


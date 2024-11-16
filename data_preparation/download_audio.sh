#!/bin/bash

# Ensure yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "yt-dlp could not be found. Please install yt-dlp first."
    exit 1
fi

## Check if the user provided the TSV file and download directory
#if [ $# -ne 2 ]; then
#    echo "Usage: $0 <path_to_tsv_file> <download_directory>"
#    exit 1
#fi

# Input TSV file and download directory
TSV_FILE="../data/index.tsv"
DOWNLOAD_DIR="../data/raw/audio_raw"

# Ensure the download directory exists
if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo "The directory $DOWNLOAD_DIR does not exist. Creating it..."
    mkdir -p "$DOWNLOAD_DIR"
fi

# Read the TSV file, skip the header, and process each line
# Assuming the TSV file is tab-separated with columns Index, Title, ID, Language
tail -n +2 "$TSV_FILE" | while IFS=$'\t' read -r Index Title ID Language; do
    # Check if the ID is a valid non-empty string
    if [ -z "$ID" ]; then
        continue
    fi

    # Construct the URL (if only the ID is provided)
    if [[ "$ID" != http* ]]; then
        URL="https://www.youtube.com/watch?v=$ID"
    else
        URL="$ID"
    fi

    # Inform the user about the video being processed
    echo "Downloading audio for: $URL"

    # Download the audio as MP3 using yt-dlp and name the file as {Index}.mp3
    yt-dlp -x --audio-format mp3 -o "$DOWNLOAD_DIR/$Index.%(ext)s" "$URL"

    # Optional: You can add a sleep time to avoid overwhelming YouTube servers
    sleep 1
done

echo "All downloads complete!"

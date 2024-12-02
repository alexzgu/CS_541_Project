#!/bin/bash

# Function to rename files in a given directory
rename_files() {
  local dir="\$1"

  # Loop through all .mp3 files in the specified directory
  for file in "$dir"/*.mp3; do
    # Check if the file exists to avoid errors
    if [[ -f "$file" ]]; then
      # Extract the integer part (before the underscore)
      integer="${file##*/}"  # Get the filename without the path
      integer="${integer%%_*}"  # Get the part before the first underscore

      # Construct the new filename
      new_name="${dir}/${integer}.mp3"

      # Rename the file
      mv "$file" "$new_name"
      echo "Renamed: $file -> $new_name"
    fi
  done
}

# Rename files in both subdirectories
rename_files "vocals"
rename_files "instrumentals"

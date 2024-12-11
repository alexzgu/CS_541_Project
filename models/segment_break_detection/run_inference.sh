#!/bin/bash

#########################################################################
# Script Name: run_inference.sh
# Description: This script processes input audio and CSV files to generate
#              input.npy and then performs model inference using the
#              TemporalCNN model.
# Usage:       ./run_inference.sh --input_dir /path/to/input_directory \
#                                 [--model_path /path/to/best_model.pth] \
#                                 [--output_activation] \
#                                 [--use_gpu]
#########################################################################

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage
usage() {
    echo "Usage: \$0 --input_dir /path/to/input_directory [--model_path /path/to/best_model.pth] [--output_activation] [--use_gpu]"
    echo ""
    echo "Options:"
    echo "  --input_dir           Path to the directory containing input audio MP3 and CSV files. (Required)"
    echo "  --model_path          Path to the saved model weights. Default is 'best_model.pth'."
    echo "  --output_activation   Include this flag to apply sigmoid activation to model outputs."
    echo "  --use_gpu             Include this flag to use GPU for inference if available."
    echo "  --help                Display this help message."
    exit 1
}

# Default values
MODEL_PATH="best_model.pth"
OUTPUT_ACTIVATION=false
USE_GPU=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case \$1 in
        --input_dir)
            INPUT_DIR="\$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="\$2"
            shift 2
            ;;
        --output_activation)
            OUTPUT_ACTIVATION=true
            shift 1
            ;;
        --use_gpu)
            USE_GPU=true
            shift 1
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown parameter passed: \$1"
            usage
            ;;
    esac
done

# Check if input_dir is provided
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input_dir is required."
    usage
fi

# Check if input_dir exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' does not exist."
    exit 1
fi

# Define paths
AUDIO_FILE=""
CSV_FILE=""

# Find audio MP3 and CSV files in the input directory
for file in "$INPUT_DIR"/*; do
    if [[ "$file" == *.mp3 ]]; then
        AUDIO_FILE="$file"
    elif [[ "$file" == *.csv ]]; then
        CSV_FILE="$file"
    fi
done

# Validate that both files are found
if [ -z "$AUDIO_FILE" ]; then
    echo "Error: No MP3 audio file found in '$INPUT_DIR'."
    exit 1
fi

if [ -z "$CSV_FILE" ]; then
    echo "Error: No CSV file found in '$INPUT_DIR'."
    exit 1
fi

echo "Found audio file: $AUDIO_FILE"
echo "Found CSV file: $CSV_FILE"

# Define the path for input.npy
INPUT_NPY="$INPUT_DIR/input.npy"

# ===============================
# Step 1: Preprocess Input Data
# ===============================
echo "Step 1: Preprocessing input data to generate input.npy..."

# Call the preprocessing script
python preprocess.py --audio "$AUDIO_FILE" --csv "$CSV_FILE" --output "$INPUT_NPY" --debug

echo "Preprocessing completed. Generated input.npy at '$INPUT_NPY'."

# ==================================
# Step 2: Perform Model Inference
# ==================================
echo "Step 2: Performing model inference..."

# Check if input.npy exists
if [ ! -f "$INPUT_NPY" ]; then
    echo "Error: input.npy not found at '$INPUT_NPY'. Ensure that preprocessing was successful."
    exit 1
fi

# Build the command to run model_inference.py
INFERENCE_CMD="python model_inference.py --model_path \"$MODEL_PATH\" --input \"$INPUT_NPY\""

if $OUTPUT_ACTIVATION; then
    INFERENCE_CMD+=" --output_activation"
fi

if $USE_GPU; then
    INFERENCE_CMD+=" --use_gpu"
fi

# Execute the inference command
echo "Running inference command:"
echo "$INFERENCE_CMD"
eval $INFERENCE_CMD

echo "Inference completed successfully."

#########################################################################
# End of run_inference.sh
#########################################################################

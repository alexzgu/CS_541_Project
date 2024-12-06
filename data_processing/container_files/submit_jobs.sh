#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Create log and output directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# writing permissions
chmod +x "$HYPER_TUNE_SLURM"
chmod +x "$TESTING_DEF"

apptainer build "$CONTAINER_SIF" "$TESTING_DEF"

learning_rates=(0.1 0.5 1.0)

job_ids=()
for lr in "${learning_rates[@]}"; do
    job_id=$(sbatch --parsable "$HYPER_TUNE_SLURM" $lr)
    job_ids+=($job_id)
done

echo "All jobs submitted. Use 'squeue -u $USER' to check their status."
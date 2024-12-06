#!/bin/bash

#SBATCH -N 1              # Request 1 node for the job
#SBATCH -n 1              # Request 1 CPU core
#SBATCH --mem=2g          # Request 2 GB of memory
#SBATCH -p short          # Use the "short" partition
#SBATCH -t 02:00:00       # Set the maximum time to 2 hours
#SBATCH -J hyper_tune     # Job name prefix
#SBATCH -o /home/azgu/container_files/log/slurm_%j.out   # Output file
#SBATCH -e /home/azgu/container_files/log/slurm_%j.err   # Error file

# Load environment variables
set -a
source .env
set +a

# check if learning rate is provided
if [ -z "$1" ]; then
    echo "Error: Learning rate not provided"
    exit 1
fi

LEARNING_RATE=$1

# run the container
apptainer run --userns --bind "$OUTPUT_DIR:/output" "$CONTAINER_SIF" $LEARNING_RATE

echo "Job completed for learning rate: $LEARNING_RATE"
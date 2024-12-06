#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# ensure we're in the output directory
cd "$OUTPUT_DIR" || exit

# combine all individual output files
cat output_lr_*.txt > "$COMBINED_RESULTS"

echo "All results have been combined into $COMBINED_RESULTS"

# extract and display final accuracies
echo "Final accuracies:" >> "$COMBINED_RESULTS"
grep "Best accuracy achieved:" output_lr_*.txt >> "$COMBINED_RESULTS"

# optional: Clean up individual output files
# uncomment the following line if you want to remove individual output files
# rm output_lr_*.txt

echo "Results collection completed."
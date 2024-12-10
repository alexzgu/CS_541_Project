import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.segment_break_detection.utils.dataset import BreakDataset


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# Initialize dataset (adjust the paths as needed)
data_dir = '../../../data/clean'
audio_dir = f'{data_dir}/audio/vocals'
label_dir = f'{data_dir}/segment_breaks'

# Parameters
interval_width = 20  # ms per interval
seconds_per_image = 5
intervals_per_image = int((seconds_per_image * 1000) / interval_width)

dataset = BreakDataset(audio_dir=audio_dir, label_dir=label_dir, interval_width=interval_width)

# Create the 'test_images_uwu' directory
save_dir = 'test_images_uwu'
ensure_dir(save_dir)

# Iterate over the dataset and save 5-second interval images
for idx in range(len(dataset)):
    file_idx, image_slices, labels = dataset[idx]

    # Iterate through the image slices in 5-second chunks
    for start in range(0, len(image_slices), intervals_per_image):
        # Get the end index for this 5-second chunk
        end = min(start + intervals_per_image, len(image_slices))

        # Extract the image slices for this chunk
        chunk_image_slices = image_slices[start:end]
        chunk_labels = labels[start:end]

        # Create a filename for each 5-second chunk
        chunk_filename = f"{file_idx}_from_{start * interval_width / 1000:.2f}s_to_{end * interval_width / 1000:.2f}s.png"
        chunk_image_path = os.path.join(save_dir, chunk_filename)

        # If the chunk is less than 5 seconds, pad with zeros
        if chunk_image_slices.shape[0] < intervals_per_image:
            padding = torch.zeros(
                intervals_per_image - chunk_image_slices.shape[0],
                chunk_image_slices.shape[1],
                chunk_image_slices.shape[2]
            )
            chunk_image_slices = torch.cat([chunk_image_slices, padding], dim=0)

        # Combine multiple image slices vertically
        # Reshape and normalize for saving
        combined_image = chunk_image_slices.permute(1, 0, 2)  # [width, num_intervals, height]

        # Convert to numpy and scale to 0-255
        combined_image_np = combined_image.numpy()
        combined_image_np = ((combined_image_np - combined_image_np.min()) /
                             (combined_image_np.max() - combined_image_np.min()) * 255).astype(np.uint8)

        # Create and save PIL Image
        pil_image = Image.fromarray(combined_image_np, mode='L')  # 'L' for grayscale
        pil_image.save(chunk_image_path)

        # Optional: Print additional information
        print(f"Saved 5-second chunk: {chunk_filename}")
        print(f"Chunk time range: {start * interval_width / 1000:.2f}s to {end * interval_width / 1000:.2f}s")
        print(f"Number of intervals in this chunk: {chunk_image_slices.shape[0]}")
        print(f"Breaks in this chunk: {chunk_labels.sum().item()}")
        print("---")

    # Uncomment the break if you want to process only the first file
    break

import os
import subprocess


def replace_audio(video_dir, audio_dir):
    # Ensure input directories exist
    if not os.path.exists(video_dir) or not os.path.exists(audio_dir):
        raise ValueError("One or both input directories do not exist")

    # Get all .webm files
    webm_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.webm')])

    for webm_file in webm_files:
        # Get the base number from filename
        base_num = webm_file.replace('.webm', '')

        # Construct corresponding mp3 filename
        mp3_file = f"{base_num}.mp3"
        mp3_path = os.path.join(audio_dir, mp3_file)

        # Check if corresponding mp3 exists
        if not os.path.exists(mp3_path):
            print(f"Warning: No matching audio file found for {webm_file}")
            continue

        webm_path = os.path.join(video_dir, webm_file)
        output_path = os.path.join(video_dir, f"temp_{webm_file}")

        # Use ffmpeg to replace audio
        try:
            cmd = [
                'ffmpeg',
                '-i', webm_path,  # Input video
                '-i', mp3_path,  # Input audio
                '-c:v', 'copy',  # Copy video codec
                '-map', '0:v',  # Use video from first input
                '-map', '1:a',  # Use audio from second input
                output_path
            ]
            subprocess.run(cmd, check=True)

            # Replace original file with new version
            os.replace(output_path, webm_path)
            print(f"Successfully processed {webm_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error processing {webm_file}: {e}")

data_dir = "../../data"
videos_dir = f"{data_dir}/processed/manually_edited_srts"
vocals_dir = f"{data_dir}/clean/audio/vocals"
replace_audio(videos_dir, vocals_dir)
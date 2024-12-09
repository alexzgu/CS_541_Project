from pydub import AudioSegment
import math

def clip_and_pad_audio(input_mp3_path, clip_duration, interval_size, output_path):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_mp3_path)

    # Clip the audio to specified duration (converting seconds to milliseconds)
    clipped_audio = audio[:int(clip_duration * 1000)]

    # Calculate current duration in seconds
    current_duration = len(clipped_audio) / 1000

    # Round up to nearest multiple of interval_size
    rounded_duration = math.ceil(current_duration / interval_size) * interval_size

    # Calculate final duration (rounded + 5 intervals)
    final_duration = rounded_duration + (5 * interval_size)

    # Calculate how much silence to add (in milliseconds)
    silence_duration = int((final_duration - current_duration) * 1000)

    # Generate silence
    silence = AudioSegment.silent(duration=silence_duration)

    # Combine audio and silence
    final_audio = clipped_audio + silence

    # Export the result
    final_audio.export(output_path, format="mp3")

if __name__ == "__main__":
    input_dir = '../../data/clean/audio/vocals'
    output_dir = 'golden_audio'
    time_limits = [(0,-1), (6, 99.70), (16, -1), (19, 73)]
    fixed_interval = 20

    for idx, limit in time_limits:
        input_mp3_path = f'{input_dir}/{idx}.mp3'
        output_path = f'{output_dir}/{idx}.mp3'

        if limit >0:
            clip_and_pad_audio(input_mp3_path, limit, fixed_interval, output_path)
        else:
        # just make a regular copy of the mp3
        # round up to nearest multiple of 20 ms
            audio = AudioSegment.from_mp3(input_mp3_path)
            rounded_duration = math.ceil(len(audio) / 1000 / fixed_interval) * fixed_interval
            audio.export(output_path, format="mp3")

def progress_count(current, total):
    ending = '\n' if current == total else ''
    print(f'\r{current}/{total}', end=ending)


def calculate_sample_number(time_ms, sampling_rate):
    return int((time_ms / 1000) * sampling_rate)
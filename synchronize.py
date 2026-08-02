import csv
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate2d, stft


DATA_FOLDER = "OnlyR"
ROOT_DIR = Path(__file__).absolute().parent / "Data" / DATA_FOLDER
VIDEO_DIR = ROOT_DIR / "videos"
MICRO_DATA_DIR = ROOT_DIR / "micro_data"

# One convention for every device:
# delay = device_time - camera_time
# sync_timestamp = timestamp - delay
DOBOT_CAMERA_DELAY_S = -0.080667
USG_CAMERA_DELAY_S = 0.028167
IMU_CAMERA_DELAY_S = 0.015917
PSMOVE_CAMERA_DELAY_S = 0.0

AUDIO_CHANNEL = 0
VIDEO_AUDIO_CHANNEL = 0


def read_wave(path):
    sample_rate, x = wavfile.read(path)
    x = x.T
    if x.dtype == np.int32:
        x = x / float(2**31 - 1)
    elif x.dtype == np.int16:
        x = x / float(2**15 - 1)
    if len(x.shape) == 1:
        x = x[None, :]
    return sample_rate, x


def generate_chirp_signal(
    duration=0.2,
    start_freq=500,
    end_freq=4000,
    sample_rate=44100,
):
    t = np.linspace(
        0,
        duration,
        int(duration * sample_rate),
        endpoint=False,
    )
    chirp_signal = np.sin(
        2
        * np.pi
        * np.interp(t, [0, duration], [start_freq, end_freq])
        * t
    )
    return chirp_signal


def extract_audio_from_video(video_file):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_wav_file = os.path.join(
            tmpdirname,
            os.path.basename(video_file) + ".wav",
        )
        ffmpeg_command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_file,
            "-vn",
            tmp_wav_file,
        ]
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not os.path.exists(tmp_wav_file):
            raise RuntimeError(
                f"Could not extract audio from {video_file}. "
                "The video probably has no audio stream."
            )
        return read_wave(tmp_wav_file)


def has_audio_stream(video_file):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_file),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def calculate_energy_with_stft(signal, fs, n_fft=2048):
    signal = signal - signal.mean()
    f, t, zxx = stft(signal, fs, window="hann", nperseg=n_fft)
    magnitude = np.abs(zxx)
    energy = np.sum(magnitude ** 2, axis=1)
    return f, t, magnitude, energy


def sync_spectrograms(ref, measured):
    if ref.shape == measured.shape:
        return 0

    ref = ref > np.max(ref) * 0.8
    ref = ref.astype(np.float32)
    ref = ref - ref.mean()
    corr = correlate2d(
        ref,
        np.log10(measured + 1e-10),
        "valid",
    ).squeeze()
    return len(corr) - np.argmax(corr)


def argmax_correlation(input_signal, sync_signal, fs, n_fft=1024):
    _, input_times, input_spectrum, _ = calculate_energy_with_stft(
        input_signal,
        fs,
        n_fft,
    )
    _, _, sync_spectrum, _ = calculate_energy_with_stft(
        sync_signal,
        fs,
        n_fft,
    )

    sync_index = sync_spectrograms(sync_spectrum, input_spectrum)
    if sync_index >= len(input_times):
        return None

    sync_time = input_times[sync_index]
    return int(sync_time * fs)


def find_delay_by_sync(
    video_file,
    audio_file,
    audio_channel,
    video_channel=0,
):
    audio_fs, audio_signal = read_wave(audio_file)
    audio_signal = audio_signal[audio_channel, :]
    sync_signal = generate_chirp_signal(sample_rate=audio_fs)
    audio_shift = argmax_correlation(
        audio_signal,
        sync_signal,
        audio_fs,
    )

    video_fs, video_signal = extract_audio_from_video(video_file)
    video_signal = video_signal[video_channel, :]

    if audio_fs != video_fs:
        sync_signal = generate_chirp_signal(sample_rate=video_fs)

    video_shift = argmax_correlation(
        video_signal,
        sync_signal,
        video_fs,
    )

    if None in [audio_shift, video_shift]:
        return None

    return audio_shift / audio_fs - video_shift / video_fs


def recording_name(video_path):
    name = video_path.stem
    for suffix in ("_cam1", "_cam2"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def timestamp_delays():
    return {
        "dobot": DOBOT_CAMERA_DELAY_S,
        "usg_timestamps": USG_CAMERA_DELAY_S,
        "imu": IMU_CAMERA_DELAY_S,
        "psmove": PSMOVE_CAMERA_DELAY_S,
    }


def find_matching_files(folder, name):
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.stem == name
    )


def correct_csv(path, delay):
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])
        timestamp_column = next(
            (
                column
                for column in fieldnames
                if column.lower() == "timestamp"
            ),
            None,
        )

        if timestamp_column is None:
            print(f"Skipping {path}: no timestamp column")
            return

        timestamp_index = fieldnames.index(timestamp_column) + 1
        if "sync_timestamp" not in fieldnames:
            fieldnames.insert(timestamp_index, "sync_timestamp")

        rows = list(reader)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            raw_timestamp = float(row[timestamp_column])
            row["sync_timestamp"] = f"{raw_timestamp - delay:.6f}"
            writer.writerow(row)

    temporary_path.replace(path)

    print(f"CSV: {path}, delay={delay * 1000:.2f} ms")


def trim_audio_start(audio_path, video_path):
    delay = find_delay_by_sync(
        str(video_path),
        str(audio_path),
        AUDIO_CHANNEL,
        VIDEO_AUDIO_CHANNEL,
    )

    if delay is None:
        print(f"Audio unchanged: could not find chirp in {audio_path.name}")
        return

    if delay <= 0:
        print(f"Audio unchanged: {audio_path.name}, delay={delay:.6f} s")
        return

    sample_rate, samples = wavfile.read(audio_path)
    samples_to_remove = int(round(delay * sample_rate))

    if samples_to_remove >= len(samples):
        raise RuntimeError(
            f"Audio delay is longer than the recording: {audio_path}"
        )

    temporary_path = audio_path.with_name(
        audio_path.stem + "_sync_tmp" + audio_path.suffix
    )
    wavfile.write(
        temporary_path,
        sample_rate,
        samples[samples_to_remove:],
    )
    temporary_path.replace(audio_path)

    print(
        f"Audio: {audio_path.name}, removed {delay:.6f} s from start"
    )


def process_recording(name, videos, delays):
    print(f"\nRecording: {name}")

    for folder_name, delay in delays.items():
        source_dir = ROOT_DIR / folder_name

        for source in find_matching_files(source_dir, name):
            correct_csv(source, delay)

    video_with_audio = next(
        (video for video in sorted(videos) if has_audio_stream(video)),
        None,
    )
    for audio_path in find_matching_files(MICRO_DATA_DIR, name):
        if video_with_audio is None:
            print(
                f"Audio unchanged: no embedded camera audio for "
                f"{audio_path.name}"
            )
            continue
        trim_audio_start(audio_path, video_with_audio)


def main():
    delays = timestamp_delays()
    recordings = {}

    if not VIDEO_DIR.is_dir():
        raise FileNotFoundError(
            f"Video directory does not exist: {VIDEO_DIR}"
        )

    for video_path in sorted(VIDEO_DIR.iterdir()):
        if not video_path.is_file():
            continue

        name = recording_name(video_path)
        recordings.setdefault(name, []).append(video_path)

    for name in sorted(recordings):
        process_recording(name, recordings[name], delays)

    print("\nFinished.")


if __name__ == "__main__":
    main()

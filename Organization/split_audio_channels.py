import glob
import os
import subprocess
from pathlib import Path

AUDIO_FOLDER = "../audios"

if __name__ == "__main__":
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True)

    for file_path in audio_files:
        ptp_path = str(os.path.join(*os.path.dirname(file_path).split("/")[-3:]))
        base_name = Path(file_path).stem

        left = os.path.join(AUDIO_FOLDER, ptp_path, base_name + "_left.wav")
        right = os.path.join(AUDIO_FOLDER, ptp_path, base_name + "_right.wav")

        split_channels = ["ffmpeg", "-i", file_path, "-map_channel", "0.0.0", left, "-map_channel", "0.0.1", right]
        subprocess.run(split_channels)

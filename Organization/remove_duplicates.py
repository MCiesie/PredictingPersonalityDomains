import glob
import os
import subprocess
from pathlib import Path

AUDIO_FOLDER = "./normalized_audio"
TRANSCRIPT_FOLDER = "../Transcriptions"
DUPLICATES = "./combined_audios.txt"


if __name__ == "__main__":
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True)
    os.makedirs(os.path.join(AUDIO_FOLDER, "unusable"), exist_ok=True)
    os.makedirs(os.path.join(TRANSCRIPT_FOLDER, "unusable"), exist_ok=True)

    with open(DUPLICATES, "r") as f:
        duplicates = f.read()

    for file_path in audio_files:
        base_name = Path(file_path).stem
        if base_name in duplicates:
            ptp_path = str(os.path.join(*os.path.dirname(file_path).split("/")[2:]))

            move_unusable = [
                "mv", os.path.join(AUDIO_FOLDER, ptp_path, base_name + ".wav"),
                os.path.join(AUDIO_FOLDER, "unusable", base_name + ".wav"),
            ]
            subprocess.run(move_unusable)
            move_unusable = [
                "mv", os.path.join(TRANSCRIPT_FOLDER, ptp_path, base_name + ".json"),
                os.path.join(TRANSCRIPT_FOLDER, "unusable", base_name + ".json"),
            ]
            subprocess.run(move_unusable)

import glob
import os
import subprocess
from pathlib import Path

AUDIO_FOLDER = "../audios"
TRANSCRIPT_FOLDER = "../Transcriptions"

if __name__ == "__main__":
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True)

    for file_path in audio_files:
        ptp_path = str(os.path.join(*os.path.dirname(file_path).split("/")[-3:]))
        base_name = Path(file_path).stem
        transcript_path = os.path.join(TRANSCRIPT_FOLDER, ptp_path, base_name + ".json")
        unusable_path = os.path.join(TRANSCRIPT_FOLDER, "unusable", base_name + ".json")

        if os.path.exists(unusable_path):
            move_transcript = ["mv", unusable_path, transcript_path]
            subprocess.run(move_transcript)

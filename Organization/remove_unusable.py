import glob
import os
import re
import subprocess
from pathlib import Path

AUDIO_FOLDER = "../audios"
REMOVE_PATHS = "./inaudible.txt"
NORMALIZED_FOLDER = "./normalized_audio"
TRANSCRIPT_FOLDER = "../Transcriptions"

def remove_if_unusable(input_path):
    if "2020-01-23_PTP1470_9. Sitzung" in input_path:
        return input_path

    print(f"Analyzing {input_path}...")
    analyze_cmd = [
        "ffmpeg", "-i", input_path,
        "-af", "ebur128=peak=true",
        "-f", "null", "-"
    ]
    result = subprocess.run(analyze_cmd, capture_output=True, text=True)

    lines = result.stderr.splitlines()
    loudness_range = False
    for line in lines:
        if "Loudness range:" in line:
            loudness_range = True
        if loudness_range and "LRA:" in line:
            match = re.search(r"LRA:\s+([0-9.]+)", line)
            if match:
                lra_value = float(match.group(1))
                print(f"LRA for {input_path}: {lra_value}")
                if lra_value == 0.0:
                    return input_path
    return None


if __name__ == "__main__":
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True)
    os.makedirs(os.path.join(NORMALIZED_FOLDER, "unusable"), exist_ok=True)
    os.makedirs(os.path.join(TRANSCRIPT_FOLDER, "unusable"), exist_ok=True)

    with open(REMOVE_PATHS, "w") as f:

        for file_path in audio_files:
            path = remove_if_unusable(file_path)
            if path is not None:
                f.write(path + "\n")

                ptp_path = str(os.path.join(*os.path.dirname(file_path).split("/")[2:]))
                base_name = Path(file_path).stem

                move_unusable = [
                    "mv", os.path.join(NORMALIZED_FOLDER, ptp_path, base_name + ".wav"), os.path.join(NORMALIZED_FOLDER, "unusable", base_name + ".wav"),
                ]
                subprocess.run(move_unusable)
                move_unusable = [
                    "mv", os.path.join(TRANSCRIPT_FOLDER, ptp_path, base_name + ".json"), os.path.join(TRANSCRIPT_FOLDER, "unusable", base_name + ".json"),
                ]
                subprocess.run(move_unusable)

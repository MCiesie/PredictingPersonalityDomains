import os
import glob
import subprocess
from pathlib import Path

input_dir = "../audios"
output_dir = "./normalized_audio"
os.makedirs(output_dir, exist_ok=True)

audio_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

for file_path in audio_files:
    filename = Path(file_path).name
    output_path = os.path.join(output_dir, filename)

    # Analyze loudness
    analyze_cmd = [
        "ffmpeg", "-i", file_path,
        "-af", "loudnorm=I=-14:TP=0.0:LRA=7.0:print_format=summary",
        "-f", "null", "-"
    ]
    result = subprocess.run(analyze_cmd, capture_output=True, text=True)

    lines = result.stderr.splitlines()
    current_lufs = None
    for line in lines:
        if "Input Integrated" in line:
            current_lufs = float(line.split(":")[-1].strip().replace(" LUFS", ""))
            break

    if current_lufs is None:
        print("Could not measure LUFS, skipping.")
        continue

    # Increase volume if necessary
    if current_lufs < -35:
        increase_db = 40
        intermediate_file = "temp_boosted.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", file_path,
            "-af", f"volume={increase_db}dB",
            intermediate_file
        ])
        input_for_normalization = intermediate_file
    else:
        input_for_normalization = file_path

    # Normalize volume
    normalize_cmd = [
        "ffmpeg", "-y", "-i", input_for_normalization,
        "-af", "loudnorm=I=-14:TP=0.0:LRA=7.0",
        "-ar", "16000", "-ac", "1",
        output_path
    ]
    subprocess.run(normalize_cmd)

    # Delete temp file
    if os.path.exists("temp_boosted.wav"):
        os.remove("temp_boosted.wav")

print("\nAll files processed.")

import glob
import os
import subprocess

AUDIO_FOLDER = "./normalized_audio"

if __name__ == "__main__":
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True)

    for file_path in audio_files:
        if "combined" in file_path:
            remove_audio = ["rm", file_path]
            subprocess.run(remove_audio)

import glob
import os
import pandas as pd
from pathlib import Path
import torch
from prepare_data import extract_pid, extract_session

AUDIO_FEATURES_FOLDER = "./audio_features_new"
DUPLICATES = "./combined_audios.txt"
OUTPUT_FILE = "./all_audio_data.pt"


if __name__ == "__main__":
    with open(DUPLICATES, "r") as f:
        duplicates = f.read()

    features_files = glob.glob(os.path.join(AUDIO_FEATURES_FOLDER, "**", "*.csv"), recursive=True)

    all_data = []

    for file_path in features_files:
        filename = Path(file_path).stem
        if filename in duplicates:
            continue

        audio_dict = {}

        df = pd.read_csv(file_path)
        df = df.drop(columns=["file", "start", "end"], errors="ignore")
        features = torch.tensor(df.values, dtype=torch.float32)

        audio_dict["PID"] = extract_pid(file_path)
        audio_dict["session"] = extract_session(filename)
        audio_dict["features"] = features

        all_data.append(audio_dict)

    # Save to file
    torch.save(all_data, OUTPUT_FILE)

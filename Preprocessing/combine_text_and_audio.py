import glob
import json
import os

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from prepare_data import extract_pid, extract_session

def to_seconds(td_str):
    return pd.to_timedelta(td_str).total_seconds()

def load_audio_features_map(audio_features_path):
    audio_data = glob.glob(os.path.join(audio_features_path, "**", "*.csv"), recursive=True)

    # Create a fast lookup dictionary
    audio_lookup = {}
    for file_path in audio_data:
        filename = Path(file_path).stem
        session = extract_session(filename)

        if 1 not in session and 8 not in session:
            continue

        session_num = 1 if 1 in session else 8
        pid = extract_pid(file_path)

        key = (pid, session_num)
        audio_lookup[key] = file_path

    return audio_lookup

def extract_turn_features(turn_times, audio_df, feature_cols, fill_value=0.0):
    # Convert to numpy
    starts = audio_df['start_sec'].to_numpy()
    ends = audio_df['end_sec'].to_numpy()
    features = audio_df[feature_cols].to_numpy()

    num_features = features.shape[1]
    out = np.full((len(turn_times), num_features * 2), fill_value, dtype=np.float32)

    for i, (t_start, t_end) in enumerate(turn_times):
        mask = (starts >= t_start) & (ends <= t_end)
        if np.any(mask):
            seg = features[mask]
            mean_feats = np.nanmean(seg, axis=0)
            std_feats = np.nanstd(seg, axis=0)
            out[i, :num_features] = mean_feats
            out[i, num_features:] = std_feats

    return out  # shape: (num_turns, num_features*2)

def prepare_dataset(
    transcript_data,
    opensmile_data,
    embedding_dir,
    fill_value=0.0
):
    all_text_features = []
    all_audio_features = []
    all_labels = []
    all_metadata = []

    with open(transcript_data, 'r', encoding='utf-8') as f:
        transcript_files = json.load(f)

    opensmile_data = load_audio_features_map(opensmile_data)

    for session_data in transcript_files:
        if 1 not in session_data["session"] and 8 not in session_data["session"]:
            continue

        pid = session_data["PID"]
        session_num = 1 if 1 in session_data["session"] else 8

        # Load text embeddings
        emb_path = Path(embedding_dir) / f"{pid}_session{session_num}_embeddings.npy"
        text_embeddings = np.load(emb_path)

        # Load audio features
        csv_path = opensmile_data.get((pid, session_num))
        if csv_path is None:
            print(f"{pid} session {session_num} not found in opensmile data")
            continue
        audio_df = pd.read_csv(csv_path)
        audio_df['start_sec'] = audio_df['start'].apply(to_seconds)
        audio_df['end_sec'] = audio_df['end'].apply(to_seconds)
        feature_cols = [
            col for col in audio_df.columns
            if col not in ['file', 'start', 'end', 'start_sec', 'end_sec']
        ]

        # Vectorized turn-level audio features
        turn_times = session_data["time"]
        turn_audio_features = extract_turn_features(
            turn_times, audio_df, feature_cols, fill_value=fill_value
        )

        all_text_features.append(text_embeddings)
        all_audio_features.append(turn_audio_features)
        labels = session_data["labels_pre"] if session_num == 1 else session_data["labels_post"]
        all_labels.append(labels)
        all_metadata.append({"PID": pid, "session": session_num, "speaker": session_data["speaker"]})

        if np.isnan(labels).any():
            print(f"NaN labels for {pid}")

    return {
        "text": all_text_features,
        "audio": all_audio_features,
        "labels": all_labels,
        "metadata": all_metadata
    }


if __name__ == "__main__":
    dataset = prepare_dataset(
        transcript_data="./all_data_patient.json", # NEW
        opensmile_data="./audio_features_new",
        embedding_dir="./text_embeddings_patient" # NEW
    )

    print(len(dataset["text"]))
    print(dataset["text"][0].shape, dataset["audio"][0].shape)
    torch.save(dataset, "processed_dataset.pt")
    print("Dataset saved successfully")



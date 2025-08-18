import chardet
import glob
import json
import numpy as np
import os
import pandas as pd
import re
from collections import defaultdict
from pathlib import Path

# File locations
TRANSCRIPT_FOLDER = "../Transcriptions"
DUPLICATES = "./combined_audios.txt"
#AUDIO_FEATURES_FOLDER = "./audio_features"
LABELS_DATA = "../share_Mateusz/dat_TUD.csv"
PT_TO_PTP_MAPPING = "../MPI_Psychiatrie/codebooks/PT to PTP.xlsx"
SPEAKER_MAP = "./speaker_map.json"
OUTPUT_FILE = "./all_data.json"


def detect_encoding(labels_path):
    with open(labels_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]


def load_labels():
    encoding = detect_encoding(LABELS_DATA)
    return pd.read_csv(LABELS_DATA, encoding=encoding, sep=';')


def load_mapping():
    return pd.read_excel(PT_TO_PTP_MAPPING, sheet_name='Tabelle1')


def extract_pid(file_path):
    match = re.search(r'(PTP?\d{4})', file_path)
    if match:
        return str(match.group(1))
    raise ValueError(f"No PTP ID found in {file_path}")


def extract_session(base_name):
    pattern = re.compile(
        r'(?:Sitzung|Sitzuing|Situng)[\s._-]*(1[0-5]|[1-9])(?:\s*\+\s*(1[0-5]|[1-9]))*'
        r'|(?:^|[\s._-])(1[0-5]|[1-9])(?:\s*\+\s*(1[0-5]|[1-9]))*[\s._-]*(?:Sitzung|Sitzuing|Situng)',
        re.IGNORECASE
    )
    match = pattern.search(base_name)
    if not match:
        return ["Follow-up / Zoom"]

    groups = [g for g in match.groups() if g is not None]
    session_nums = [int(g) for g in groups if g.isdigit()]
    if max(session_nums) > 15:
        raise ValueError(f"Corrupted session number found in {base_name}")
    return session_nums


def load_speaker_map():
    with open(SPEAKER_MAP, "r") as f:
        return json.load(f)


def process_transcript(transcript_path, speaker_map, all_labels, mapping):
    with open(transcript_path, "r") as f:
        transcript = json.load(f)

    base_name = Path(file_path).stem
    trans_dict = defaultdict(list)

    pid = extract_pid(file_path)

    trans_dict["PID"] = pid
    trans_dict["session"] = extract_session(base_name)

    trans_dict["turns"] = []
    trans_dict["speaker"] = []
    trans_dict["time"] = []
    last_texts = []

    for seg in transcript["segments"]:
        text = seg.get("text")
        if text in last_texts[-2:]:  # Check for last two repeats
            continue
        trans_dict["turns"].append(text)
        speaker_id = seg.get("speaker")
        speaker_name = speaker_map.get(base_name).get(speaker_id, "Unknown")
        trans_dict["speaker"].append(speaker_name)
        trans_dict["time"].append((seg.get("start"), seg.get("end")))
        last_texts.append(text)

    curr_row = all_labels[all_labels["ID"] == pid]
    if curr_row.empty:
        try:
            mapped_pid = mapping[mapping["PT"] == pid]["PTP"].values[0]
            curr_row = all_labels[all_labels["ID"] == mapped_pid]
        except IndexError:
            print(f"No label row for {pid}: {transcript_path}")
            return None

    facets = get_facets(curr_row)
    trans_dict["labels_pre"] = [float(x.iloc[0]) for x in facets[0] + facets[2]]
    trans_dict["labels_post"] = [float(x.iloc[0]) for x in facets[1] + facets[3]]

    return trans_dict


def get_facets(patient):
    pid5_domains = [
        [patient["t0_pid_negaff"],
        patient["t0_pid_detach"],
        patient["t0_pid_antago"],
        patient["t0_pid_disinh"],
        patient["t0_pid_psycho"]],

        [patient["t7_pid_negaff"],
        patient["t7_pid_detach"],
        patient["t7_pid_antago"],
        patient["t7_pid_disinh"],
        patient["t7_pid_psycho"]]
    ]

    # LPFS domain scores need to be calculated manually
    def precompute_lpfs(patient, post_treatment=False):
        lpfs_items = [patient[f"t{int(post_treatment)*7}_lpfs_{i}"] for i in range(1, 81)]
        identity_scores = [lpfs_items[item_idx] for item_idx in [3, 6, 7, 14, 22, 32, 34, 39, 46, 47, 49, 54, 55, 56, 61, 64, 65, 70, 71, 73, 75, 76, 78]]
        identity_weights = [-0.5, 2.5, 1.5, 3.5, 1.5, 2.5, 2.5, -0.5, 3.5, 0.5, 1.5, 2.5, 2.5, 3.5, 3.5, 0.5, 2.5, 3.5, 2.5, 0.5, -0.5, 1.5, 1.5]
        self_dir_scores = [lpfs_items[item_idx] for item_idx in [0, 13, 15, 21, 23, 24, 27, 28, 29, 33, 35, 40, 41, 50, 52, 57, 58, 63, 69, 77, 79]]
        self_dir_weights = [-0.5, -0.5, 3.5, 3.5, 2.5, 2.5, 0.5, 0.5, 3.5, 0.5, 1.5, 1.5, 2.5, -0.5, 2.5, 3.5, 3.5, 0.5, 1.5, 1.5, 0.5]
        empathy_scores = [lpfs_items[item_idx] for item_idx in [1, 10, 16, 17, 19, 20, 30, 31, 37, 43, 48, 53, 59, 60, 67, 74]]
        empathy_weights = [2.5, -0.5, 2.5, 2.5, 1.5, 3.5, 1.5, 3.5, 1.5, -0.5, 3.5, 2.5, 0.5, -0.5, 0.5, 0.5]
        intimacy_scores = [lpfs_items[item_idx] for item_idx in [2, 4, 5, 8, 9, 11, 12, 18, 25, 26, 36, 38, 42, 44, 45, 51, 62, 66, 68, 72]]
        intimacy_weights = [2.5, 2.5, 0.5, 3.5, 3.5, 1.5, 1.5, 3.5, -0.5, 1.5, -0.5, -0.5, 3.5, 2.5, 2.5, 0.5, 3.5, 0.5, 3.5, 2.5]

        return [sum([score * weight for score, weight in zip(identity_scores, identity_weights)]),
                sum([score * weight for score, weight in zip(self_dir_scores, self_dir_weights)]),
                sum([score * weight for score, weight in zip(empathy_scores, empathy_weights)]),
                sum([score * weight for score, weight in zip(intimacy_scores, intimacy_weights)])]

    lpfs_domains = [precompute_lpfs(patient, False), precompute_lpfs(patient, True)]

    return pid5_domains + lpfs_domains


if __name__ == "__main__":
    transcript_files = glob.glob(os.path.join(TRANSCRIPT_FOLDER, "**", "*.json"), recursive=True)
    all_labels = load_labels()
    mapping = load_mapping()
    speaker_map = load_speaker_map()
    with open(DUPLICATES, "r") as f:
        duplicates = f.read()

    all_data = []

    for file_path in transcript_files:
        if Path(file_path).stem in duplicates:
            continue

        processed = process_transcript(file_path, speaker_map, all_labels, mapping)
        if processed:
            if 1 in processed["session"] or 8 in processed["session"]:
                all_data.append(processed)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

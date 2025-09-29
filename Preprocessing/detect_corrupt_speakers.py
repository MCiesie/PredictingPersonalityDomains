import glob
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

TRANSCRIPT_FOLDER = "../Transcriptions"
DUPLICATES = "./combined_audios.txt"

def guess_speaker_roles(segments):
    speaker_questions = {"SPEAKER_00": 0, "SPEAKER_01": 0}
    speaker_segments_total = {"SPEAKER_00": 0, "SPEAKER_01": 0}
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        if "?" in seg["text"]:
            speaker_questions[speaker] += 1

        speaker_segments_total[speaker] += 1

    speaker_ratio = speaker_segments_total["SPEAKER_00"] / sum(speaker_segments_total.values())
    if speaker_ratio < 0.15 or speaker_ratio > 0.85:
        return speaker_ratio

    sorted_speakers = sorted(speaker_questions.items(), key=lambda x: x[1], reverse=True)
    therapist, patient = sorted_speakers[0][0], sorted_speakers[1][0]
    return {patient: "Patient", therapist: "Therapist"}

def get_speaker_ratio(segments):
    speaker_segments_total = {"SPEAKER_00": 0, "SPEAKER_01": 0}
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            continue

        speaker_segments_total[speaker] += 1

    speaker_ratio = max(speaker_segments_total["SPEAKER_00"], speaker_segments_total["SPEAKER_01"]) / sum(speaker_segments_total.values())

    return speaker_ratio


if __name__ == "__main__":
    speaker_map_json = "speaker_map.json"
    transcript_files = glob.glob(os.path.join(TRANSCRIPT_FOLDER, "**", "*.json"), recursive=True)
    with open(DUPLICATES, "r") as f:
        duplicates = f.read()
    ratio_list = []

    with open("corrupt_speakers.txt", "w") as output:
        for file_path in transcript_files:
            if Path(file_path).stem in duplicates:
                continue
            
            with open(file_path, 'r', encoding='utf-8') as input:
                data = json.load(input)
    
            segments = data.get("segments")
            ratio_list.append(get_speaker_ratio(segments))
            try:
                speaker_map = guess_speaker_roles(segments)
            except ZeroDivisionError:
                print("Division by zero: " + file_path)
                speaker_map = {}
    
            if type(speaker_map) != dict:
                output.write(file_path + f": Speaker_00 has {speaker_map * 100:.2f}% of segments\n")
                speaker_map = {}
    
            session_id = os.path.splitext(os.path.basename(file_path))[0]
            if os.path.exists(speaker_map_json):
                with open(speaker_map_json, 'r') as map_f:
                    all_maps = json.load(map_f)
            else:
                all_maps = {}
            all_maps[session_id] = speaker_map
            with open(speaker_map_json, 'w') as map_f:
                json.dump(all_maps, map_f, indent=2)

    plt.figure(figsize=(8, 5))
    sns.histplot(ratio_list, bins=10, kde=False)

    plt.xlabel("Max speaker segment ratio per file")
    plt.ylabel("Number of files")
    plt.title("Distribution of dominant speaker ratios across sessions")
    plt.xticks([i / 10 for i in range(5, 11)])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speaker_ratio_plot.png", dpi=300, bbox_inches='tight')


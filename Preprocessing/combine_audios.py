import glob
import os
import pandas as pd
from pydub import AudioSegment
import re
from pathlib import Path

AUDIO_FOLDER = "./normalized_audio"
COMBINED_FILES = "./combined_audios.txt"


def extract_pid(file_path):
    match = re.search(r'(PTP?\d{4})', file_path)
    if match:
        return str(match.group(1))
    raise ValueError(f"No PTP ID found in {file_path}")


def extract_session(base_name):
    pattern = re.compile(
        r'(?:Sitzung|Sitzuing|Situng)[\s._-]*([1-9]|1[0-4])(?:\s*\+\s*([1-9]|1[0-4]))*'
        r'|([1-9]|1[0-4])(?:\s*\+\s*([1-9]|1[0-4]))*[\s._-]*(?:Sitzung|Sitzuing|Situng)',
        re.IGNORECASE
    )
    match = pattern.search(base_name)
    if not match:
        return "Follow-Up"

    groups = [g for g in match.groups() if g is not None]
    session_nums = [int(g) for g in groups if g.isdigit()]
    if max(session_nums) > 14:
        raise ValueError(f"Corrupted session number found in {base_name}")
    return min(session_nums)

def get_audio_length(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio)
    except:
        return None

def extract_audio_info(file_path):
    return {"PID": extract_pid(file_path), "session": extract_session(Path(file_path).stem),
            "audio_length": get_audio_length(file_path), "file_path": str(file_path)}


if __name__ == "__main__":
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True)
    df = pd.DataFrame([extract_audio_info(f) for f in audio_files])

    grouped = df.groupby(["PID", "session"], group_keys=False)
    df_sorted = grouped.apply(lambda g: g.sort_values("audio_length", ascending=False))

    with open(COMBINED_FILES, "w") as f:
        for (pid, sid), group in df_sorted.groupby(["PID", "session"]):
            if group.shape[0] < 2:
                continue
    
            # Combine audio
            f.write("Now combining:\n")
            combined = AudioSegment.empty()
            for _, row in group.iterrows():
                path = row["file_path"]
                audio = AudioSegment.from_file(path)
                combined += audio
                f.write(f"{path}\n")
            f.write("\n")
    
            # Save combined file
            dir_path = Path(path).parent
            output_path = dir_path / f"{pid}_Sitzung{sid}_combined.wav"
            combined.export(output_path, format="wav")
            print(f"Saved: {output_path}")

import glob
import os
from pathlib import Path
import opensmile

from prepare_data import extract_session

AUDIO_FOLDER = "./normalized_audio"
FEATURES_FOLDER = "./audio_features_new"
REDO = False

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True)

for file_path in audio_files:
    if "unusable" in file_path:
        continue

    file_path = Path(file_path)
    base_name = file_path.stem

    session = extract_session(base_name)
    if 1 not in session and 8 not in session:
        continue

    ptp_path = Path(*file_path.parent.parts[1:])

    (Path(FEATURES_FOLDER) / ptp_path).mkdir(parents=True, exist_ok=True)

    features_path = Path(FEATURES_FOLDER) / ptp_path / f"{base_name}.csv"

    if os.path.exists(features_path) and not REDO:
        continue

    feat_df = smile.process_file(file_path)
    feat_df.to_csv(features_path)

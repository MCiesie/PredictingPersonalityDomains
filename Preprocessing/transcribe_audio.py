import os
import glob
import subprocess
from pathlib import Path

import whisperx
import torch
import json

# Settings
AUDIO_FOLDER = "../audios"
NORMALIZED_AUDIO = "./normalized_audio"
OUTPUT_FOLDER = "../Transcriptions"
LANGUAGE = "de"
MODEL_SIZE = "medium"
DIARIZE = True
REDO = True

def normalize_audio(input_path, output_path):
    # Analyze loudness
    analyze_cmd = [
        "ffmpeg", "-i", input_path,
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
        return

    # Increase volume if necessary
    if current_lufs < -35:
        increase_db = 40
        intermediate_file = "temp_boosted.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"volume={increase_db}dB",
            intermediate_file
        ])
        input_for_normalization = intermediate_file
    else:
        input_for_normalization = input_path

    # Normalize volume
    normalize_cmd = [
        "ffmpeg", "-y", "-i", input_for_normalization,
        "-af", "loudnorm=I=-14:TP=0.0:LRA=7.0",
        "-ar", "16000", "-ac", "1",
        output_path
    ]
    #normalize_cmd = [
    #    "ffmpeg", "-i", input_for_normalization,
    #    "-filter:a", "loudnorm", output_path
    #]
    subprocess.run(normalize_cmd)

    # Delete temp file
    if os.path.exists("temp_boosted.wav"):
        os.remove("temp_boosted.wav")


def transcribe_audio(model, audio_path, transcript_path, batch_size, device, diarize_model):
    # Transcribe
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Align timestamps
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Assign speakers
    if diarize_model:
        diarize_segments = diarize_model(audio, num_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    # Save to output
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nFinished")


if __name__ == "__main__":
    os.makedirs(NORMALIZED_AUDIO, exist_ok=True)
    # Load WhisperX model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    print(f"Loading WhisperX model on {device}...")
    model = whisperx.load_model(MODEL_SIZE, device, language=LANGUAGE)

    # Diarize output
    diarize_model = None
    if DIARIZE and device == "cuda":
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="",
                                                             device=device)

    # Get all .wav files recursively
    audio_files = glob.glob(os.path.join(NORMALIZED_AUDIO, "**", "*.wav"), recursive=True)

    for file_path in audio_files:
        if not "combined" in file_path:
            continue

        file_path = Path(file_path)
        ptp_path = Path(*file_path.parent.parts[1:])

        (Path(OUTPUT_FOLDER) / ptp_path).mkdir(parents=True, exist_ok=True)
        (Path(NORMALIZED_AUDIO) / ptp_path).mkdir(parents=True, exist_ok=True)

        base_name = file_path.stem
        normalized_path = Path(NORMALIZED_AUDIO) / ptp_path / f"{base_name}.wav"
        transcript_path = Path(OUTPUT_FOLDER) / ptp_path / f"{base_name}.json"

        # Skip if already transcribed
        if os.path.exists(transcript_path) and os.path.exists(normalized_path) and not REDO:
            continue

        # Normalize volume and transcribe
        try:
            print("\nNormalizing volume ...")
            normalize_audio(file_path, normalized_path)
            print(f"\nTranscribing: {base_name}")
            transcribe_audio(model, file_path, transcript_path, batch_size, device, diarize_model)

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

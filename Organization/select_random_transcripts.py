import glob
import os
import random

AUDIO_FOLDER = "../audios"
OUTPUT_FILE = "./random_selection.txt"
TO_CHOOSE = 15

audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True)
NUMBER_OF_TRANSCRIPTS = len(audio_files)

my_selection = random.sample(range(NUMBER_OF_TRANSCRIPTS), TO_CHOOSE)

transcript_id = 0
with open(OUTPUT_FILE, "w") as f:
    f.write(f"Choosing {TO_CHOOSE} out of {NUMBER_OF_TRANSCRIPTS} random transcripts:\n")
    for audio_path in audio_files:
        if transcript_id in my_selection:
            f.write(f"{audio_path}\n")

        transcript_id += 1
        if transcript_id > max(my_selection):
            exit()

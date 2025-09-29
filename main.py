import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from Repo.PredictingPersonalityDomains.data.dataset import TherapyDataset
from milintsevich_2023.model import PID5SymptomPredictor
from ray_2019.model import MultimodalPID5Model

from Repo.PredictingPersonalityDomains.training.loop import train, test
from Repo.PredictingPersonalityDomains.utils.baselines import calculate_baseline_mean, calculate_baseline_majority
from Repo.PredictingPersonalityDomains.utils.split import shuffle_data, subset_data

parser=argparse.ArgumentParser()
parser.add_argument("model", choices=['MultimodalPersonalityModel', 'PID5SymptomPredictor', 'MultimodalPID5Model'])
parser.add_argument("mode", choices=['Regression', 'Classification'])
args=parser.parse_args()

# File locations
ALL_DATA = "./training_dataset.pt"
MODEL = eval(args.model)
DATASET = TherapyDataset
MODE = str(args.mode) # "Regression" or "Classification"
FUSION = "crossmodal" # "crossmodal" or "simple"

EPOCHS = 200
BATCH_SIZE = 10
PATIENCE = 10

LR = 5e-4
SEEDS = [42, 123, 456, 789, 999]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    all_data = torch.load("training_dataset_patient.pt")
    with open(f"./metadata_patient.json", "r") as f:
        metadata = json.load(f)

    n = all_data["text"].shape[0]
    print(f"Number of files: {n}")

    train_idx, val_idx, test_idx = shuffle_data(metadata)

    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    assert set(train_idx) & set(val_idx) == set()
    assert set(train_idx) & set(test_idx) == set()
    assert set(val_idx) & set(test_idx) == set()
    
    train_data = subset_data(all_data, train_idx)
    val_data = subset_data(all_data, val_idx)
    test_data = subset_data(all_data, test_idx)

    audio_mean = np.nanmean(train_data["audio"], axis=0)
    audio_std = np.nanstd(train_data["audio"], axis=0)

    train_dataset = DATASET(train_data, audio_means=audio_mean, audio_stds=audio_std, setup=MODE.lower())

    label_means, label_stds = train_dataset.means, train_dataset.stds

    val_dataset = DATASET(val_data, label_means, label_stds, audio_mean, audio_std, setup=MODE.lower())
    test_dataset = DATASET(test_data, label_means, label_stds, audio_mean, audio_std, setup=MODE.lower())

    if MODE == "Regression":
        print("Calculating mean baseline...")
        calculate_baseline_mean(train_dataset, test_dataset)
    else:
        print("Calculating majority class baseline...")
        calculate_baseline_majority(train_dataset, test_dataset)

    all_scores = []
    for seed in SEEDS:
        model_state, best_val_score = train(seed, train_dataset, val_dataset)
        print(f"Best Val 'MAE' for seed {seed}: {best_val_score:.4f}")
        train_score = test(model_state, train_dataset, seed)
        print(f"Train 'MAE' for seed {seed}: {train_score:.4f}")
        test_score = test(model_state, test_dataset, seed)
        print(f"Test 'MAE' for seed {seed}: {test_score:.4f}")
        all_scores.append(test_score)
    print(f"Average Test 'MAE' over seeds: {np.mean(all_scores):.4f}")


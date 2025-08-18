import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from Milintsevich_2023.model import PID5SymptomPredictor
from Ray_2019.model import MultimodalPID5Model
from dataset import TherapyDataset

# File locations
ALL_DATA = "./training_dataset.pt"
MODEL = PID5SymptomPredictor

EPOCHS = 200
BATCH_SIZE = 8

LR = 3e-5
LOSS = nn.MSELoss #nn.SmoothL1Loss
SEEDS = [42, 123, 456, 789, 999]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_EVERY = 3
LOAD_MODEL = False
LOAD_FROM_EPOCH = 0


# Evaluation
def compute_metrics(preds, targets):
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Remove NaNs
    mask = ~np.isnan(preds_np) & ~np.isnan(targets_np)
    preds_np = preds_np[mask]
    targets_np = targets_np[mask]

    if len(preds_np) == 0:
        print("Warning: NaNs only")
        return {"mse": np.nan, "mae": np.nan, "r2": np.nan}

    mse = mean_squared_error(targets_np, preds_np)
    mae = mean_absolute_error(targets_np, preds_np)
    r2 = r2_score(targets_np, preds_np)

    return {"mse": mse, "mae": mae, "r2": r2}


# Training loop
def train(seed, train_data, val_data):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MODEL().to(DEVICE)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = LOSS()

    best_val_mae = float("inf")
    best_model = None
    start_epoch = 0

    if LOAD_MODEL:
        checkpoint = torch.load(f"./milintsevich_2023/saved_models/checkpoint_{LOAD_FROM_EPOCH}.pth.tar")
        best_model, best_val_mae, start_epoch = load_checkpoint(checkpoint, model, optimizer)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            text = batch["text"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            preds = model(text, audio, mask)

            #print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                text = batch["text"].to(DEVICE)
                audio = batch["audio"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                preds = model(text, audio, mask)
                loss = criterion(preds, labels)
                total_val_loss += loss.item()

                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_loader)
        preds_tensor = torch.cat(all_preds, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        #print("Sample preds:", preds_tensor[:5])
        #print("Sample targets:", targets_tensor[:5])

        print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}")
        metrics = compute_metrics(preds_tensor, targets_tensor)
        print(
            f"Seed {seed}, Epoch {epoch}, MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")

        val_mae = metrics["mae"]

        # Save best model
        if best_model is None or val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model.state_dict()

        # Save checkpoint every 3 epochs
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_model': best_model,
                'val_mae': best_val_mae,
                'epoch': epoch
            }
            save_checkpoint(checkpoint, metrics, epoch)

    return best_model, best_val_mae


def test(model_state, test_data, seed):
    model = MODEL().to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    all_preds, all_targets = [], []
    prediction_log = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            text = batch["text"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            preds = model(text, audio, mask)

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(preds_tensor, targets_tensor)
    #print("Preds shape:", preds_tensor.shape)
    #print("Targets shape:", targets_tensor.shape)
    prediction_log.append({"MSE": metrics["mse"], "MAE": metrics["mae"], "R²": metrics["r2"]})

    with open(f"test_predictions_seed{seed}.json", "w") as f:
        json.dump(prediction_log, f, indent=2)

    return metrics["mae"]


def save_checkpoint(state, metrics, epoch):
    print("=> Saving checkpoint")
    torch.save(state, f"./milintsevich_2023/saved_models/checkpoint_{epoch}.pth.tar")
    with open(f"./milintsevich_2023/val_metrics/epoch_{epoch}.json", "w") as f:
        json.dump(metrics, f)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['best_model'], checkpoint['best_val_mae'], checkpoint['epoch']


if __name__ == "__main__":
    all_data = torch.load("training_dataset.pt")

    n = all_data["text"].shape[0]
    print(f"Number of patients: {n}")

    # Create shuffled indices
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)

    # data split ratio akin to paper
    train_end = int(0.57 * n)
    val_end = int(0.76 * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Helper function to subset tensors by index
    def subset_data(data_dict, idx_list):
        return {k: v[idx_list] for k, v in data_dict.items()}

    train_data = subset_data(all_data, train_idx)
    val_data = subset_data(all_data, val_idx)
    test_data = subset_data(all_data, test_idx)

    train_dataset = TherapyDataset(train_data)
    val_dataset = TherapyDataset(val_data)
    test_dataset = TherapyDataset(test_data)

    y_train = train_dataset.labels.numpy()
    y_test = test_dataset.labels.numpy()

    # Mean PID-5 scores as baseline
    train_means = np.mean(y_train, axis=0)
    y_pred_baseline = np.tile(train_means, (len(y_test), 1))
    mask = ~np.isnan(y_test).any(axis=1)  # rows without NaNs

    print("Total test samples:", len(y_test))
    print("Valid samples after masking:", mask.sum())

    baseline_mae = mean_absolute_error(y_test[mask], y_pred_baseline[mask])
    print("Baseline MAE:", baseline_mae)

    per_domain_mae = np.mean(np.abs(y_test[mask] - y_pred_baseline[mask]), axis=0)
    print("Per-domain baseline MAE:", per_domain_mae)

    all_maes = []
    for seed in SEEDS:
        model_state, best_val_mae = train(seed, train_dataset, val_dataset)
        print(f"Best MAE for seed {seed}: {best_val_mae:.4f}")
        mae = test(model_state, test_dataset, seed)
        print(f"MAE for seed {seed}: {mae:.4f}")
        all_maes.append(mae)
    print(f"Average MAE over seeds: {np.mean(all_maes):.4f}")

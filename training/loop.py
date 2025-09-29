import copy
import json

import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from training.metrics import compute_regression_metrics, compute_classification_metrics
from training.losses import classification_loss, compute_class_weights
from tqdm import tqdm

from Repo.PredictingPersonalityDomains.main import MODEL, FUSION, MODE, DEVICE, BATCH_SIZE, LR, EPOCHS, args, PATIENCE
from Repo.PredictingPersonalityDomains.training.loss import compute_loss
from Repo.PredictingPersonalityDomains.utils.ordinal_classes import decode_ordinal_preds


# Training loop
def train(seed, train_data, val_data):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MODEL(FUSION, MODE.lower()).to(DEVICE)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    if args.model == "MultimodalPersonalityModel":
        text_encoder_params = list(model.text_encoder.named_parameters())
        audio_encoder_params = list(model.audio_encoder.named_parameters())
        fusion_params = list(model.fusion.named_parameters())
        head_params = list(model.head.named_parameters())

        encoder_params = [p for n, p in audio_encoder_params]
        encoder_params += [p for _, p in text_encoder_params]

        fusion_params = [p for _, p in fusion_params]
        head_params = [p for _, p in head_params]

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": 5e-4},
            {"params": fusion_params, "lr": 5e-4},
            {"params": head_params, "lr": 5e-3},
        ], weight_decay=1e-5)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    if MODE == "Classification":
        print("Using classification loss")
        # train_labels = train_data.labels.numpy()
        # facet_weights = compute_class_weights(train_labels, num_classes=4)
        # weights_avg = sum(facet_weights) / len(facet_weights)
        # facet_weights = compute_imboll_weights(train_labels, num_classes=NUM_CLASSES, device=DEVICE)

        # criterion = classification_loss
        # criterion = imb_oll_loss
        criterion = nn.BCEWithLogitsLoss()
        compute_metrics = compute_classification_metrics
        best_val_score = float("inf")
    else:
        print("Using regression loss")
        criterion = nn.MSELoss()
        compute_metrics = compute_regression_metrics
        best_val_score = float("inf")
        # facet_weights = None

    best_model = None
    patience = PATIENCE
    counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        total_samples = 0

        if epoch == 3:
            for g in optimizer.param_groups:
                if g.get("name") == "head":
                    g["lr"] = 1e-4

        for batch_idx, batch in enumerate(train_loader):
            text = batch["text"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            preds = model(text, audio, mask)

            label_mask = labels >= 0
            preds_valid, labels_valid = preds[label_mask], labels[label_mask]
            assert preds_valid.shape == labels_valid.shape

            if label_mask.any():
                loss = compute_loss(criterion, preds_valid, labels_valid, MODE)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item() * preds_valid.numel()
                total_samples += preds_valid.numel()

        avg_train_loss = total_train_loss / max(1, total_samples)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        total_samples = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"):
                text = batch["text"].to(DEVICE)
                audio = batch["audio"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                preds = model(text, audio, mask)

                label_mask = labels >= 0
                preds_valid, labels_valid = preds[label_mask], labels[label_mask]
                assert preds_valid.shape == labels_valid.shape

                if label_mask.any():
                    loss = compute_loss(criterion, preds_valid, labels_valid, MODE)

                    total_val_loss += loss.item() * labels_valid.size(0)
                    total_samples += labels_valid.size(0)

                    preds = decode_ordinal_preds(preds) if MODE == "Classification" else preds
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        avg_val_loss = total_val_loss / max(1, total_samples)
        print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}")

        preds_tensor = torch.cat(all_preds, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(preds_tensor, targets_tensor, train_data.means, train_data.stds)

        if MODE == "Classification":
            val_score = metrics["mae"]
            print(f"Epoch {epoch + 1}: MAE = {val_score}")
        else:
            val_score = metrics["mae"]

        # Save best model
        if best_model is None or val_score < best_val_score:
            best_val_score = val_score
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return best_model, best_val_score


def test(model_state, test_data, seed):
    model = MODEL(FUSION, MODE.lower()).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    all_preds, all_targets = [], []
    prediction_log = []

    if MODE == "Classification":
        compute_metrics = compute_classification_metrics
    else:
        compute_metrics = compute_regression_metrics

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
    metrics = compute_metrics(preds_tensor, targets_tensor, test_data.means, test_data.stds)

    if MODE == "Regression":
        prediction_log.append({"MSE": metrics["mse"], "MAE": metrics["mae"], "R²": metrics["r2"], "CCC": metrics["ccc"],
                               "MAE per domain": metrics["per_domain_mae"],
                               "CCC per domain": metrics["per_domain_ccc"]})
    else:
        prediction_log.append({"Accuracy": metrics["accuracy"], "Accuracy per domain": metrics["per_facet_accuracy"],
                               "MAE": metrics["mae"], "F1 score per domain": metrics["per_facet_f1"],
                               "MAE per domain": metrics["per_facet_mae"], "QWK per domain": metrics["per_facet_qwk"]})

    with open(f"test_predictions/{args.model}_test_predictions_seed{seed}_{MODE}.json", "w") as f:
        json.dump(prediction_log, f, indent=2)

    return metrics["mae"]

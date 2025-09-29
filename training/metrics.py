import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score


# Evaluation
def compute_regression_metrics(preds, targets, means=None, stds=None, baseline=False):
    """
    preds: torch.Tensor, shape (B, F, C) logits
    targets: torch.Tensor, shape (B, F) integer labels, padded positions = -1
    """
    # Rescale predictions and targets to original value range
    preds = preds * stds + means
    targets = targets * stds + means

    preds = preds.clamp(0.0, 3.0)

    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Mask invalid values (NaNs)
    mask = targets_np >= 0
    if not np.any(mask):
        print("Warning: NaNs only")
        return {
            "mse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "ccc": np.nan,
            "per_domain_mae": [],
            "per_domain_ccc": []}

    preds_valid = preds_np[mask]
    targets_valid = targets_np[mask]

    # Global metrics
    mse = mean_squared_error(targets_valid, preds_valid)
    mae = mean_absolute_error(targets_valid, preds_valid)
    r2 = r2_score(targets_valid, preds_valid)
    ccc = concordance_ccc(preds_valid, targets_valid)

    # Per-domain metrics
    num_facets = targets_np.shape[1]
    per_domain_mae, per_domain_ccc = [], []

    for i in range(num_facets):
        facet_mask = mask[:, i]
        if np.any(facet_mask):
            preds_i, targets_i = preds_np[facet_mask, i], targets_np[facet_mask, i]
            per_domain_mae.append(float(np.mean(np.abs(preds_i - targets_i))))
            per_domain_ccc.append(float(concordance_ccc(preds_i, targets_i)))
        else:
            per_domain_mae.append(None)
            per_domain_ccc.append(None)

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "ccc": float(ccc),
        "per_domain_mae": per_domain_mae,
        "per_domain_ccc": per_domain_ccc
    }


def compute_classification_metrics(preds, targets, baseline=False):
    """
    preds: torch.Tensor, shape (B, F, C) logits
    targets: torch.Tensor, shape (B, F) integer labels, padded positions = -1
    """

    # Convert logits to predicted classes
    if not baseline:
        probs = torch.sigmoid(preds)  # probability each threshold is passed
        preds_classes = torch.sum(probs > 0.5, dim=-1)
    else:
        preds_classes = preds  # already discrete labels

    preds_classes = preds_classes.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Mask padded positions
    mask = targets_np >= 0

    if not np.any(mask):
        print("Warning: all targets are padded")
        return {
            "accuracy": np.nan,
            "per_facet_accuracy": [np.nan] * targets_np.shape[1],
            "per_facet_f1": [np.nan] * targets_np.shape[1],
            "per_facet_mae": [np.nan] * targets_np.shape[1],
            "per_facet_qwk": [np.nan] * targets_np.shape[1]
        }

    # Apply mask
    preds_valid = preds_classes[mask]
    targets_valid = targets_np[mask]

    # Global accuracy
    accuracy = accuracy_score(targets_valid, preds_valid)
    mae = mean_absolute_error(targets_valid, preds_valid)

    # Per-facet metrics
    num_facets = targets_np.shape[1]
    per_facet_accuracy = []
    per_facet_f1 = []
    per_facet_mae = []
    qwk_list = []

    for i in range(num_facets):
        facet_mask = mask[:, i]
        if np.any(facet_mask):
            facet_preds = preds_classes[:, i][facet_mask]
            facet_targets = targets_np[:, i][facet_mask]

            per_facet_accuracy.append(accuracy_score(facet_targets, facet_preds))
            per_facet_f1.append(f1_score(facet_targets, facet_preds, average="macro"))
            per_facet_mae.append(mean_absolute_error(facet_targets, facet_preds))

            try:
                qwk = cohen_kappa_score(facet_targets, facet_preds, weights="quadratic")
            except Exception as e:
                qwk = np.nan
                print("Warning:", e)
            qwk_list.append(qwk)

        else:
            per_facet_accuracy.append(np.nan)
            per_facet_f1.append(np.nan)
            qwk_list.append(np.nan)

    return {
        "accuracy": accuracy,
        "mae": mae,
        "per_facet_accuracy": per_facet_accuracy,
        "per_facet_f1": per_facet_f1,
        "per_facet_mae": per_facet_mae,
        "per_facet_qwk": qwk_list
    }


def concordance_ccc(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_var, y_var = np.var(x), np.var(y)
    covariance = np.mean((x - x_mean) * (y - y_mean))
    return (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2 + 1e-8)

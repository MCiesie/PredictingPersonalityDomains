import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

from Repo.PredictingPersonalityDomains.utils.ordinal_classes import expand_to_ordinal_targets
from Restructure.main import DEVICE


def compute_loss(criterion, preds_valid, labels_valid, mode):
    if mode == "Regression":
        return criterion(preds_valid, labels_valid)
    else:
        ordinal_targets = expand_to_ordinal_targets(labels_valid, num_classes=preds_valid.size(-1))
        return criterion(preds_valid, ordinal_targets)


def classification_loss(logits, targets, mask=None, weights=None):
    # logits: (B, F, C)
    # targets: (B, F)
    B, F, C = logits.shape
    logits = logits.view(B * F, C)
    targets = targets.view(B * F)

    if mask is not None:
        mask = mask.view(B * F)
        logits = logits[mask]
        targets = targets[mask]

    return nn.CrossEntropyLoss(weights=weights)(logits, targets) if weights is not None else nn.CrossEntropyLoss()(logits, targets)


def imb_oll_loss(logits, targets, mask=None, facet_weights=None):
    """
    logits: (B, F, C-1)
    targets: (B, F) int labels
    mask: (B, F)
    facet_weights: (F, C-1)
    """
    B, F, C_minus_1 = logits.shape
    num_classes = C_minus_1 + 1

    # convert labels to ordinal binary
    ord_targets = expand_to_ordinal_targets(targets, num_classes)  # (B, F, C-1)

    # flatten
    logits = logits.reshape(B * F, C_minus_1)
    ord_targets = ord_targets.reshape(B * F, C_minus_1)

    if mask is not None:
        mask = mask.reshape(B * F)
        logits = logits[mask]
        ord_targets = ord_targets[mask]

    if facet_weights is not None:
        facet_weights = facet_weights.unsqueeze(0).expand(B, F, C_minus_1)  # (B,F,C-1)
        facet_weights = facet_weights.reshape(B * F, C_minus_1)
        if mask is not None:
            facet_weights = facet_weights[mask]
        loss = functional.binary_cross_entropy_with_logits(
            logits, ord_targets, weight=facet_weights
        )
    else:
        loss = functional.binary_cross_entropy_with_logits(logits, ord_targets)

    return loss


def compute_imboll_weights(train_labels, num_classes, device="cpu"):
    """
    y_train: numpy array (N, F) with integer labels (may include -1 for missing)
    returns: torch.FloatTensor (F, C-1)
    """
    num_facets = train_labels.shape[1]
    weights = []

    for f in range(num_facets):
        facet_labels = train_labels[:, f]
        facet_labels = facet_labels[facet_labels >= 0]  # ignore missing

        if len(facet_labels) == 0:
            # uniform weights if no data
            w = torch.ones(num_classes - 1, device=device)
        else:
            # compute distribution of ordinal thresholds
            counts = []
            for c in range(num_classes - 1):
                # how many samples exceed threshold c?
                counts.append((facet_labels > c).sum())
            counts = np.array(counts, dtype=np.float32)
            print(f"Facet {f} class distribution:", counts)

            # imbalance weights = inverse frequency
            inv_freq = 1.0 / (counts + 1e-6)
            w = torch.tensor(inv_freq / inv_freq.sum(), dtype=torch.float, device=device)

        weights.append(w)

    return torch.stack(weights, dim=0) # (F, C-1)


def compute_class_weights(train_labels, num_classes, device=DEVICE):
    num_facets = train_labels.shape[1]
    facet_weights = []

    for i in range(num_facets):
        facet_labels = train_labels[:, i]
        facet_labels = facet_labels[~np.isnan(facet_labels)].astype(int)

        if len(facet_labels) == 0:
            # no labels for this facet
            weights = torch.ones(num_classes, device=device)
        else:
            counts = np.bincount(facet_labels, minlength=num_classes)
            print(f"Facet {i} class distribution:", counts)
            inv_freq = 1.0 / (counts + 1e-6)  # avoid div by zero
            weights = inv_freq / inv_freq.sum()  # normalize

        facet_weights.append(torch.tensor(weights, dtype=torch.float, device=device))

    return facet_weights

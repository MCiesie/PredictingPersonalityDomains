import torch

def expand_to_ordinal_targets(labels, num_classes):
    """
    labels: (N,) int64, values in [0, num_classes]
    returns: (N, num_classes) float, 0/1 targets
    """
    # Example: class = 3 → [1,1,1,0,0,...,0]
    N = labels.size(0)
    targets = torch.zeros((N, num_classes), device=labels.device)
    for i in range(num_classes):
        targets[:, i] = (labels > i).float()
    return targets


def decode_ordinal_preds(logits):
    """
    logits: (N, F, num_classes) raw logits
    returns: (N, F) predicted class index
    """
    probs = torch.sigmoid(logits)  # convert to probs
    # Count how many thresholds the model thinks are passed
    return (probs > 0.5).sum(dim=-1)
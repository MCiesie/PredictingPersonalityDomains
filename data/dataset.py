import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 20
MAX_CLASS = NUM_CLASSES - 1

NUM_FACETS = 5


class TherapyDataset(Dataset):
    def __init__(self, data_dict, means=None, stds=None, audio_means=None, audio_stds=None, num_labels=NUM_FACETS, setup="regression"):
        """
        data_dict: {
            "text":   Tensor [N, max_turns, text_dim],
            "audio":  Tensor [N, max_turns, audio_dim],
            "labels": Tensor [N, num_labels],
            "mask":   Tensor [N, max_turns]
        }
        """

        # Take only the first 5 or 9 labels (PID-5 + LPFS)
        labels = data_dict["labels"][:, :num_labels]

        # Filter: keep entries where at least one label is not NaN
        valid_mask = ~torch.all(torch.isnan(labels), dim=1)
        self.labels = labels[valid_mask]
        self.text = data_dict["text"][valid_mask]
        self.audio = data_dict["audio"][valid_mask]
        self.mask = data_dict["mask"][valid_mask]

        print("Number of valid sessions:", self.labels.size(0))

        self.means, self.stds = None, None

        if audio_means is not None and audio_stds is not None:
            # Normalize audio features
            self.audio = (self.audio - audio_means) / audio_stds
        else:
            print("Audio inconsistency")

        if setup == "classification":
            mask = torch.isnan(self.labels)
            self.labels[:, :5] = self.value_to_class(self.labels[:, :5], 0, 3)  # PID-5
            #self.labels[:, 5:9] = self.value_to_class(labels[:, 5:9], -2, 14)  # LPFS

            # Round labels to nearest integer
            x_round = torch.floor(self.labels + 0.5)
            x_round[mask] = -1
            self.labels = x_round.to(torch.long)

        if setup == "regression":
            # Compute per-label mean & std
            mask = self.labels >= 0

            if means is None or stds is None:
                labels_masked = torch.where(mask, self.labels, torch.zeros_like(self.labels))
                count = mask.sum(dim=0)

                self.means = labels_masked.sum(dim=0) / count
                sq_sum = (labels_masked ** 2).sum(dim=0)
                var = (sq_sum / count) - (self.means ** 2)
                self.stds = torch.sqrt(torch.clamp(var, min=1e-8))
            else:
                self.means, self.stds = means, stds

            # Normalize labels
            self.labels = (self.labels - self.means) / self.stds
            self.labels = torch.where(mask, self.labels, torch.full_like(self.labels, float(-1)))


    def __len__(self):
        return self.text.size(0) # N sessions

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return {
            "text": self.text[idx], # [max_turns, text_dim]
            "audio": self.audio[idx], # [max_turns, audio_dim]
            "labels": self.labels[idx], # [num_labels]
            "mask": self.mask[idx] # [max_turns]
        }

    def value_to_class(self, x, vmin, vmax):
        """
        Convert a label x into integer class 0..MAX_CLASS.
        """
        if vmax <= vmin:
            raise ValueError("vmax must be > vmin")

        # normalize to [0,1]
        norm = (x - vmin) / (vmax - vmin)
        norm = torch.clamp(norm, 0.0, 1.0)

        # map to discrete classes
        cls = torch.round(norm * MAX_CLASS).long()
        cls = torch.clamp(cls, 0, MAX_CLASS)

        return cls


def class_to_value(class_index, vmin, vmax):
    """
    Convert class index (0..MAX_CLASS) back to original continuous value.
    """
    if not (0 <= class_index <= MAX_CLASS):
        raise ValueError("class_index out of range")
    frac = class_index.float() / MAX_CLASS
    return vmin + frac * (vmax - vmin)

import torch
from torch.utils.data import Dataset


class TherapyDataset(Dataset):
    def __init__(self, data_dict):
        """
        data_dict: {
            "text":   Tensor [N, max_turns, text_dim],
            "audio":  Tensor [N, max_turns, audio_dim],
            "labels": Tensor [N, num_labels],
            "mask":   Tensor [N, max_turns]
        }
        """
        self.text = data_dict["text"]
        self.audio = data_dict["audio"]
        self.labels = data_dict["labels"][:, :5]
        self.mask = data_dict["mask"]

    def __len__(self):
        return self.text.size(0)  # N sessions

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],  # [max_turns, text_dim]
            "audio": self.audio[idx],  # [max_turns, audio_dim]
            "labels": self.labels[idx],  # [num_labels]
            "mask": self.mask[idx]  # [max_turns]
        }


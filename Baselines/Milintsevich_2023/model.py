import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# Model configuration
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
HIDDEN_DIM = 300
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, mask=None):
        # lstm_out: (batch_size, seq_len, hidden_dim)
        attn_scores = self.attn_weights(lstm_out).squeeze(-1)  # (batch_size, seq_len)

        if mask is not None:
            # mask: (batch_size, seq_len), 1 = keep, 0 = pad
            # set pad positions to very negative so softmax returns 0
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=1)           # (batch_size, seq_len)
        weighted_sum = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)
        return weighted_sum

class PID5SymptomPredictor(nn.Module):
    def __init__(self, text_input_dim=768, lstm_hidden_dim=300, num_symptoms=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=text_input_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        self.attn = AttentionLayer(lstm_hidden_dim * 2)

        self.fc1 = nn.Linear(lstm_hidden_dim * 2, 128)
        self.norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, num_symptoms)

    def forward(self, text_input, audio_input=None, mask=None):
        # text_input: (batch_size, num_turns, turn_input_dim)
        lstm_out, _ = self.lstm(text_input)               # (batch_size, num_turns, hidden_dim*2)
        session_embedding = self.attn(lstm_out, mask)               # (batch_size, hidden_dim*2)

        x = F.leaky_relu(self.fc1(session_embedding))         # (batch_size, 128)
        x = self.norm(x)
        output = self.fc2(x)                                  # (batch_size, num_symptoms)
        return output

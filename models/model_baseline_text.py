import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# Model configuration
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)
        self.attn_dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, hidden_dim)
        attn_scores = self.attn_weights(x).squeeze(-1) # (batch_size, seq_len)

        if mask is not None:
            # set pad positions to very negative so softmax returns 0
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(attn_scores, dim=1) # (batch_size, seq_len)
        weights = self.attn_dropout(weights)
        weighted_sum = torch.sum(x * weights.unsqueeze(-1), dim=1) # (batch_size, hidden_dim)
        return weighted_sum


class PID5SymptomPredictor(nn.Module):
    def __init__(self, text_input_dim=768, hidden_dim=HIDDEN_DIM, num_symptoms=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=text_input_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.attn = AttentionLayer(hidden_dim * 2)

        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc_dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_symptoms)

    def forward(self, text_input, audio_input=None, mask=None, setup="regression"):
        # text_input: (batch_size, num_turns, turn_input_dim)
        lstm_out, _ = self.lstm(text_input) # (batch_size, num_turns, hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        session_embedding = self.attn(lstm_out, mask) # (batch_size, hidden_dim*2)

        x = F.leaky_relu(self.fc1(session_embedding)) # (batch_size, 128)
        x = self.fc_dropout(x)
        output = self.fc2(x) # (batch_size, num_symptoms)
        return output

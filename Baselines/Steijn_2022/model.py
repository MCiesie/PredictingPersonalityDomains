import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=200):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj = nn.Linear(hidden_dim * 2, 128)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Attention
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)
        return self.proj(context)  # (batch_size, 128)


class InterpretablePredictor(nn.Module):
    def __init__(self, text_input_dim=512, hidden_dim=128, output_dim=5):
        super().__init__()
        self.text_encoder = TextEncoder(input_dim=text_input_dim)
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_input, audio_input):
        # text_input: (batch_size, seq_len, text_input_dim)
        text_feat = self.text_encoder(text_input)     # (batch_size, 128)
        hidden = F.relu(self.fc1(text_feat))
        output = self.fc2(hidden)  # (batch_size, 5)
        return output
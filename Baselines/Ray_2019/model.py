import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attention Layer for modality
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):  # x: (batch, seq_len, input_dim)
        weights = self.proj(x)  # (batch, seq_len, 1)

        if mask is not None:
            # mask: (batch_size, seq_len), 1 = keep, 0 = pad
            # set pad positions to very negative so softmax returns 0
            weights = weights.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(weights, dim=1)
        weighted = (x * weights).sum(dim=1)  # (batch, input_dim)
        return weighted, weights

# LSTM + Attention Encoder
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=200):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.attn = AttentionLayer(hidden_dim * 2)

    def forward(self, x, mask=None):  # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        context, _ = self.attn(lstm_out, mask)  # (batch, hidden_dim*2)
        return context

# Model
class MultimodalPID5Model(nn.Module):
    def __init__(self, text_input_dim=768, audio_input_dim=6373, hidden_dim=200, num_symptoms=5):
        super().__init__()
        self.text_encoder = ModalityEncoder(text_input_dim, hidden_dim)
        self.audio_encoder = ModalityEncoder(audio_input_dim, hidden_dim)

        fusion_dim = hidden_dim * 2 * 2  # text + audio
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_symptoms),  # Return PID-5 scores
        )

    def forward(self, text_input, audio_input, mask=None):
        text_context = self.text_encoder(text_input, mask)
        audio_context = self.audio_encoder(audio_input, mask)

        fused = torch.cat([text_context, audio_context], dim=1)  # (batch, fusion_dim)
        output = self.regressor(fused)  # (batch, 5)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Attention Layer for modality
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn_weights = nn.Linear(input_dim, 1)
        self.attn_dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):  # x: (batch, seq_len, input_dim)
        attn_scores = self.attn_weights(x).squeeze(-1)  # (batch, seq_len, 1)

        if mask is not None:
            # set pad positions to very negative so softmax returns 0
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(attn_scores, dim=1)

        # Catch NaNs if all tokens masked
        if torch.isnan(weights).any():
            weights = torch.zeros_like(weights)
            weights[:, 0] = 1.0  # fallback: attend to first token

        weights = self.attn_dropout(weights)
        weighted_sum = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, input_dim)
        return weighted_sum


# LSTM + Attention Encoder
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.attn = AttentionLayer(hidden_dim * 2)

    def forward(self, x, mask=None):  # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.norm(lstm_out)
        context = self.attn(lstm_out, mask)  # (batch, hidden_dim*2)
        return context


# Model
class MultimodalPID5Model(nn.Module):
    def __init__(self, text_input_dim=768, audio_input_dim=130, hidden_dim=HIDDEN_DIM, num_symptoms=5):
        super().__init__()
        self.text_encoder = ModalityEncoder(text_input_dim, hidden_dim)
        self.audio_encoder = ModalityEncoder(audio_input_dim, hidden_dim)

        fusion_dim = hidden_dim * 2 * 2 # text + audio
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, num_symptoms), # Five PID-5 scores
        )

    def forward(self, text_input, audio_input, mask=None, setup="regression"):
        text_context = self.text_encoder(text_input, mask)
        audio_context = self.audio_encoder(audio_input, mask)

        fused = torch.cat([text_context, audio_context], dim=1)  # (batch, fusion_dim)
        output = self.regressor(fused)  # (batch, 5)
        return output

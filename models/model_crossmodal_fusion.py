import torch
import torch.nn as nn
import torch.nn.functional as F

from Repo.PredictingPersonalityDomains.data.dataset import NUM_CLASSES, NUM_FACETS


# Encoders
class TextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln0 = nn.LayerNorm(2 * hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=2*hidden_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(2 * hidden_dim)
        self.pool = MaskedAttentionPooling(input_dim=2*hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.out_dim = 2 * hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, NUM_FACETS)
        )

    def forward(self, x, mask):
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.ln0(lstm_out)

        # Self-attention
        att_out, _ = self.mha(lstm_out, lstm_out, lstm_out, key_padding_mask=~mask)
        att_out = self.dropout(att_out)
        att_out = self.ln1(lstm_out + att_out)

        # Pooling
        session_repr = self.pool(att_out, mask) # (B, 2*H)

        x = F.leaky_relu(self.fc1(session_repr))
        x = self.fc_dropout(x)
        session_repr = 3 * torch.sigmoid(self.fc2(x))

        return session_repr, att_out


class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln0 = nn.LayerNorm(2 * hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=2*hidden_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(2 * hidden_dim)
        self.mha2 = nn.MultiheadAttention(embed_dim=2*hidden_dim, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(2 * hidden_dim)
        self.pool = MaskedAttentionPooling(input_dim=2*hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.out_dim = 2 * hidden_dim

    def forward(self, x, mask):
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.ln0(lstm_out)

        # Self-attention
        att_out, _ = self.mha(lstm_out, lstm_out, lstm_out, key_padding_mask=~mask)
        att_out = self.dropout(att_out)
        att_out = self.ln1(lstm_out + att_out)
        att_out2, _ = self.mha2(att_out, att_out, att_out, key_padding_mask=~mask)
        att_out2 = self.dropout(att_out)
        att_out = self.ln2(att_out + att_out2)

        # Pooling
        session_repr = self.pool(att_out, mask)  # (B, 2*H)

        return session_repr, att_out


class MaskedAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn_proj = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        attn_logits = self.attn_proj(x).squeeze(-1) # (B, T)
        attn_logits = attn_logits.masked_fill(~mask, -1e9) # mask padded tokens
        attn_weights = F.softmax(attn_logits, dim=1) # (B, T)
        session_vec = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1) # (B, D)
        return session_vec


class MaskedAttentionPoolingMulti(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn_proj = nn.Linear(input_dim, 4)

    def forward(self, x, mask):
        B, T, D = x.shape
        attn_logits = self.attn_proj(x)
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)

        session_vec = torch.bmm(attn_weights.transpose(1, 2), x) # (B, H, D)

        # flatten heads: (B, H*D)
        session_vec = session_vec.reshape(B, 4 * D)
        return session_vec


# Fusion strategies
class SimpleConcatenation(nn.Module):
    def __init__(self, input_dim, output_dim, num_facets):
        super().__init__()
        self.num_facets = num_facets
        self.concat_proj = nn.Linear(2*input_dim + output_dim, output_dim)

        # Facet embeddings
        self.facet_embeddings = nn.Parameter(torch.randn(num_facets, output_dim))

    def forward(self, text_repr, audio_repr, mask=None):
        B = text_repr.size(0)

        # Repeat session vectors across facets
        text_expand = text_repr.unsqueeze(1).repeat(1, self.num_facets, 1) # (B, num_facets, 2H)
        audio_expand = audio_repr.unsqueeze(1).repeat(1, self.num_facets, 1) # (B, num_facets, 2H)

        # Expand facet embeddings to batch
        facet_embed = self.facet_embeddings.unsqueeze(0).expand(B, -1, -1) # (B, num_facets, fusion_dim)

        # Concatenate text, audio and facet_embedding
        fused = torch.cat([text_expand, audio_expand, facet_embed], dim=-1) # (B, num_facets, 2H + 2H + fusion_dim)

        # Project to fusion dimension
        fused = self.concat_proj(fused) # (B, num_facets, fusion_dim)

        return fused


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, ff_dim, num_heads):
        super().__init__()
        self.ln0 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(ff_dim, dim),
        )
        self.dropout = nn.Dropout(0.8)

    def forward(self, x, context, mask=None):
        # Attention with pre-norm and residual
        attn_out, _ = self.cross_attn(self.ln0(x), context, context, key_padding_mask=~mask)
        attn_out = self.dropout(attn_out)
        x = x + attn_out
        ff_out = self.ff(self.ln1(x))
        ff_out = self.dropout(ff_out)
        x = x + ff_out
        return x


class CrossModalFusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_facets):
        super().__init__()
        self.num_facets = num_facets
        self.cross_attn_1to2 = CrossAttnBlock(dim=input_dim, ff_dim=4*input_dim, num_heads=4)
        self.cross_attn_2to1 = CrossAttnBlock(dim=input_dim, ff_dim=4*input_dim, num_heads=4)
        self.self_attn_1 = CrossAttnBlock(dim=input_dim, ff_dim=4*input_dim, num_heads=4)
        self.self_attn_2 = CrossAttnBlock(dim=input_dim, ff_dim=4*input_dim, num_heads=4)
        self.facet_M1_attn = CrossAttnBlock(dim=input_dim, ff_dim=4*input_dim, num_heads=4)
        self.facet_M2_attn = CrossAttnBlock(dim=input_dim, ff_dim=4*input_dim, num_heads=4)
        self.fusion_proj = nn.Linear(2*input_dim + output_dim, output_dim)

        # Facet embeddings
        self.facet_embeddings = nn.Parameter(torch.randn(num_facets, output_dim))

    def forward(self, M1, M2, mask=None):
        B = M1.size(0)

        # Cross-attention
        M1_to_M2 = self.cross_attn_1to2(M1, M2, mask)
        M2_to_M1 = self.cross_attn_2to1(M2, M1, mask)

        # Self-attention
        M1_refined = self.self_attn_1(M1_to_M2, M1_to_M2, mask)
        M2_refined = self.self_attn_2(M2_to_M1, M2_to_M1, mask)

        facet_embed = self.facet_embeddings.unsqueeze(0).expand(B, -1, -1) # (B, num_facets, fusion_dim)

        M1_facet_repr = self.facet_M1_attn(facet_embed, M1_refined, mask)
        M2_facet_repr = self.facet_M2_attn(facet_embed, M2_refined, mask)

        fused = torch.cat([M1_facet_repr, M2_facet_repr, facet_embed], dim=-1) # (B, 2*H)
        fused = self.fusion_proj(fused) # (B, D)
        fused = F.relu(fused)
        return fused


# Task head
class TaskHead(nn.Module):
    def __init__(self, input_dim, setup, num_facets, num_classes):
        super().__init__()
        self.setup = setup
        self.num_facets = num_facets
        self.num_classes = num_classes

        if setup == "classification":
            self.head = nn.Linear(input_dim, num_classes - 1)
        else:
            self.head = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, num_facets, input_dim)
        out = self.head(x) # (B, num_facets, num_classes-1) or (B, num_facets, 1)

        if self.setup == "classification":
            return out # logits per facet
        else:
            return out.squeeze(-1) # (B, num_facets)


# Full model
class MultimodalPersonalityModel(nn.Module):
    def __init__(self, fusion="crossmodal", setup="classification",
                 num_facets=NUM_FACETS, num_classes=NUM_CLASSES,
                 text_dim=768, audio_dim=130, hidden_dim=50, fusion_dim=100):
        super().__init__()
        self.setup = setup
        self.fusion_strategy = fusion

        # Encoders (B, 2*H), (B, T, 2*H)
        self.text_encoder = TextEncoder(input_dim=text_dim, hidden_dim=hidden_dim)
        self.audio_encoder = AudioEncoder(input_dim=audio_dim, hidden_dim=hidden_dim)

        # Fusion (B, num_facets, fusion_dim)
        if fusion == "simple":
            self.fusion = SimpleConcatenation(input_dim=2*hidden_dim, output_dim=fusion_dim, num_facets=num_facets)
        else:
            self.fusion = CrossModalFusion(input_dim=2*hidden_dim, output_dim=fusion_dim, num_facets=num_facets)

        # Task head (B, num_facets, num_classes) or (B, num_facets)
        self.head = TaskHead(input_dim=fusion_dim, setup=setup,
                             num_facets=num_facets, num_classes=num_classes)

    def forward(self, text_input, audio_input, mask):
        text_repr, text_seq = self.text_encoder(text_input, mask)
        audio_repr, audio_seq = self.audio_encoder(audio_input, mask)

        if self.fusion_strategy == "crossmodal":
            fused = self.fusion(text_seq, audio_seq, mask)
        else:
            fused = self.fusion(text_repr, audio_repr)

        preds = self.head(fused) # (B, num_facets, num_classes) or (B, num_facets)
        return preds

import torch
import torch.nn as nn
import math

# ─── Configuration ──────────────────────────────────────────────────────────
# Model hyperparameters (Must match training config)
SEQ_LEN = 10
NUM_FEATURES = 6
D_MODEL = 64
N_HEADS = 4
N_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.3

class PositionalEncoding(nn.Module):
    """
    Learned positional encoding for sequence positions.
    Tells the Transformer the order of cycles (cycle 1 vs cycle 10).
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.encoding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.encoding(positions)


class ChargeGPTModel(nn.Module):
    """
    Physics-Informed Transformer for Battery SOH Prediction.

    Architecture:
        Input (batch, seq_len, num_features)
            → Linear Projection to d_model
            → Positional Encoding
            → Transformer Encoder (2 layers, 4 heads)
            → Global Average Pooling
            → MLP Head → Sigmoid
        Output: SOH in [0, 1]
    """
    def __init__(self, num_features=NUM_FEATURES, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_ENCODER_LAYERS,
                 dim_ff=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()

        # Project input features to d_model dimension
        self.input_projection = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, features)
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Prediction head (MLP)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # SOH is always between 0 and 1
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, seq_len, num_features)
        returns: (batch_size, 1) — predicted SOH
        """
        # Project input features to d_model
        x = self.input_projection(x)           # (batch, seq, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)               # (batch, seq, d_model)

        # Layer norm
        x = self.layer_norm(x)                 # (batch, seq, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)        # (batch, seq, d_model)

        # Global average pooling across sequence dimension
        x = x.mean(dim=1)                      # (batch, d_model)

        # Prediction head
        x = self.prediction_head(x)            # (batch, 1)

        return x.squeeze(-1)                   # (batch,)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import pickle
import math
import warnings
warnings.filterwarnings("ignore")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Configuration ──────────────────────────────────────────────────────────

# Model hyperparameters
SEQ_LEN = 10           # Look at last 10 cycles to predict SOH
NUM_FEATURES = 6       # voltage_mean, voltage_std, current_mean, temperature_mean, discharge_capacity, c_rate
D_MODEL = 64           # Transformer hidden dimension
N_HEADS = 4            # Number of attention heads
N_ENCODER_LAYERS = 2   # Number of transformer encoder layers
DIM_FEEDFORWARD = 128  # Feedforward network dimension
DROPOUT = 0.3          # Dropout rate

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
TRAIN_SPLIT = 0.8

# Feature columns (order matters!)
FEATURE_COLS = ["voltage_mean", "voltage_std", "current_mean",
                "temperature_mean", "discharge_capacity", "c_rate"]
TARGET_COL = "soh"


# =============================================================================
# PART 1: DATA PREPARATION
# =============================================================================

class BatteryDataset(Dataset):
    """
    Creates sliding window sequences from battery cycling data.
    Each sample = (sequence of SEQ_LEN cycles, SOH at last cycle).
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(features, targets, seq_len, c_rates):
    """
    Create sliding window sequences, respecting C-rate boundaries.
    We don't want windows that cross from one C-rate group to another.
    """
    X, y = [], []
    unique_rates = np.unique(c_rates)

    for rate in unique_rates:
        # Get indices for this C-rate
        mask = c_rates == rate
        rate_features = features[mask]
        rate_targets = targets[mask]

        # Create sliding windows within this C-rate group
        for i in range(len(rate_features) - seq_len + 1):
            X.append(rate_features[i:i + seq_len])
            y.append(rate_targets[i + seq_len - 1])  # Target = SOH at last step

    return np.array(X), np.array(y)


def prepare_data(csv_path="battery_data.csv"):
    """
    Load CSV, normalize features, create train/val splits.
    Returns DataLoaders, scaler, and metadata.
    """
    print("=" * 60)
    print("  STEP 1: DATA PREPARATION")
    print("=" * 60)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"C-rates: {sorted(df['c_rate'].unique())}")

    # Extract features and target
    features = df[FEATURE_COLS].values
    targets = df[TARGET_COL].values
    c_rates = df["c_rate"].values

    # Normalize features to [0, 1]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Save scaler for inference later
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to scaler.pkl")

    # Create sliding window sequences
    X, y = create_sequences(features_scaled, targets, SEQ_LEN, c_rates)
    print(f"Created {len(X)} sequences (seq_len={SEQ_LEN})")

    # Shuffle and split
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    # Create DataLoaders
    train_dataset = BatteryDataset(X_train, y_train)
    val_dataset = BatteryDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, scaler


# =============================================================================
# PART 2: TRANSFORMER MODEL
# =============================================================================

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

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PART 3: TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    """
    Training loop with early stopping, learning rate scheduling,
    and comprehensive logging.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: TRAINING THE TRANSFORMER")
    print("=" * 60)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Epochs: {epochs} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print()

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_mae": [], "val_mae": [],
        "lr": []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # ─── Train ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_train = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * len(batch_X)
            train_mae += torch.abs(pred - batch_y).sum().item()
            n_train += len(batch_X)

        train_loss /= n_train
        train_mae /= n_train

        # ─── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = model(batch_X)
                loss = criterion(pred, batch_y)

                val_loss += loss.item() * len(batch_X)
                val_mae += torch.abs(pred - batch_y).sum().item()
                n_val += len(batch_X)

        val_loss /= n_val
        val_mae /= n_val

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                  f"LR: {current_lr:.2e}")

        # ─── Early Stopping ─────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'num_features': NUM_FEATURES,
                    'd_model': D_MODEL,
                    'n_heads': N_HEADS,
                    'n_layers': N_ENCODER_LAYERS,
                    'dim_ff': DIM_FEEDFORWARD,
                    'dropout': DROPOUT,
                    'seq_len': SEQ_LEN,
                    'feature_cols': FEATURE_COLS,
                },
                'best_val_loss': best_val_loss,
                'best_val_mae': val_mae,
                'epoch': epoch + 1,
            }, "chargegpt_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    print(f"\n✅ Best model saved to chargegpt_model.pth")
    print(f"   Best Val Loss: {best_val_loss:.6f}")
    print(f"   Best Val MAE:  {history['val_mae'][np.argmin(history['val_loss'])]:.4f}")

    return history


# =============================================================================
# PART 4: EVALUATION
# =============================================================================

def evaluate_model(model, val_loader):
    """Detailed evaluation with per-sample predictions."""
    print("\n" + "=" * 60)
    print("  STEP 3: MODEL EVALUATION")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Metrics
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    # R² score
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Max error
    max_error = np.max(np.abs(preds - targets))

    print(f"\n  📊 Evaluation Metrics:")
    print(f"  ─────────────────────────────")
    print(f"  MSE:       {mse:.6f}")
    print(f"  RMSE:      {rmse:.6f}")
    print(f"  MAE:       {mae:.4f} ({mae*100:.2f}%)")
    print(f"  R² Score:  {r2:.6f}")
    print(f"  Max Error: {max_error:.4f} ({max_error*100:.2f}%)")
    print()

    # Quality assessment
    if mae < 0.02:
        print("  ✅ EXCELLENT — MAE < 2%. Model is highly accurate.")
    elif mae < 0.05:
        print("  ✅ GOOD — MAE < 5%. Model is reliable.")
    elif mae < 0.10:
        print("  ⚠️  FAIR — MAE < 10%. Model needs tuning.")
    else:
        print("  ❌ POOR — MAE > 10%. Check data or architecture.")

    return preds, targets, {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================

def plot_training_history(history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", color="#3498db")
    axes[0].plot(history["val_loss"], label="Val Loss", color="#e74c3c")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # MAE curves
    axes[1].plot(history["train_mae"], label="Train MAE", color="#3498db")
    axes[1].plot(history["val_mae"], label="Val MAE", color="#e74c3c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("SOH Prediction Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(history["lr"], color="#2ecc71")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved training history to training_history.png")


def plot_predictions(preds, targets):
    """Plot predicted vs actual SOH."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(targets, preds, alpha=0.3, s=10, color="#3498db")
    axes[0].plot([0.5, 1.0], [0.5, 1.0], 'r--', linewidth=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual SOH")
    axes[0].set_ylabel("Predicted SOH")
    axes[0].set_title("Predicted vs Actual SOH")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # Error distribution
    errors = (preds - targets) * 100  # Convert to percentage
    axes[1].hist(errors, bins=50, color="#3498db", alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Prediction Error (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Error Distribution (Mean: {np.mean(errors):.3f}%, Std: {np.std(errors):.3f}%)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("prediction_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved prediction results to prediction_results.png")


# =============================================================================
# MAIN — RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    print("⚡ ChargeGPT — Transformer Model Training")
    print("=" * 60)

    # Step 1: Prepare data
    train_loader, val_loader, scaler = prepare_data("battery_data.csv")

    # Step 2: Create model
    model = ChargeGPTModel().to(device)
    print(f"\n🧠 Model Architecture:")
    print(model)
    print(f"\nTotal Parameters: {model.count_parameters():,}")

    # Step 3: Train
    history = train_model(model, train_loader, val_loader)

    # Step 4: Load best model and evaluate
    checkpoint = torch.load("chargegpt_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")

    preds, targets, metrics = evaluate_model(model, val_loader)

    # Step 5: Visualize
    plot_training_history(history)
    plot_predictions(preds, targets)

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("  📦 FILES TO DOWNLOAD")
    print("=" * 60)
    print("  1. chargegpt_model.pth  — Trained model weights + config")
    print("  2. scaler.pkl           — Feature scaler for inference")
    print("  3. training_history.png — Training curves")
    print("  4. prediction_results.png — Prediction accuracy plots")
    print()
    print("  Place these in your chargeGPT/models/ directory.")
    print("=" * 60)

    # Save config as JSON for reference
    config = checkpoint['config']
    config['metrics'] = {k: float(v) for k, v in metrics.items()}
    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  5. model_config.json    — Model configuration + metrics")

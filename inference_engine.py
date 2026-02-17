import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────────────────

# Model hyperparameters (MUST match training script)
SEQ_LEN = 10           # Look at last 10 cycles to predict SOH
NUM_FEATURES = 6       # voltage_mean, voltage_std, current_mean, temperature_mean, discharge_capacity, c_rate
D_MODEL = 64
N_HEADS = 4
N_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.3

# Physics Constants (for feature simulation)
NOMINAL_CAPACITY = 5.0
NOMINAL_VOLTAGE = 3.7
AMBIENT_TEMP = 25.0
THERMAL_RESISTANCE = 3.5

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "chargegpt_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Device configuration
device = torch.device("cpu") # Default to CPU for inference as it's lightweight


# =============================================================================
# PART 1: MODEL ARCHITECTURE (Copied from training script)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.encoding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.encoding(positions)


class ChargeGPTModel(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_ENCODER_LAYERS,
                 dim_ff=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.prediction_head(x)
        return x.squeeze(-1)


# =============================================================================
# PART 2: PHYSICS-BASED FEATURE GENERATION
# =============================================================================

class PhysicsSimulator:
    """
    Generates synthetic battery features based on C-rate and cycle constraints.
    These features represent the 'Physics-Informed' inputs to the Transformer.
    """
    @staticmethod
    def compute_cell_temperature(c_rate, cycle_num=1, ambient=AMBIENT_TEMP):
        heating = THERMAL_RESISTANCE * (c_rate ** 1.8)
        aging_heat = 0.002 * cycle_num * c_rate
        return round(ambient + heating + aging_heat, 2)

    @staticmethod
    def compute_voltage_features(c_rate, soh, cycle_num=1):
        degradation = 1.0 - soh
        voltage_mean = NOMINAL_VOLTAGE + 0.2 - (0.8 * degradation) - (0.05 * c_rate)
        voltage_std = 0.15 + (0.3 * degradation) + (0.02 * c_rate)
        return round(voltage_mean, 4), round(abs(voltage_std), 4)

    @staticmethod
    def compute_current_features(c_rate, soh):
        nominal_current = c_rate * NOMINAL_CAPACITY * soh
        return round(nominal_current, 4)

    @staticmethod
    def compute_discharge_capacity(soh):
        return round(max(NOMINAL_CAPACITY * soh, 0), 4)

    @staticmethod
    def generate_sequence(c_rate, current_soh, seq_len=SEQ_LEN):
        """
        Creates a sequence of features for the model input.
        Assume the battery state is roughly constant over the short window (10 cycles).
        """
        sequence = []
        for i in range(seq_len):
            # For this short window, we assume SOH is constant at 'current_soh'
            # to see what the model predicts for *this state*.
            # Or we could slightly degrade it? Let's keep it constant for simplicity.
            
            temp = PhysicsSimulator.compute_cell_temperature(c_rate)
            v_mean, v_std = PhysicsSimulator.compute_voltage_features(c_rate, current_soh)
            i_mean = PhysicsSimulator.compute_current_features(c_rate, current_soh)
            d_cap = PhysicsSimulator.compute_discharge_capacity(current_soh)
            
            # Feature order MUST match training: ["voltage_mean", "voltage_std", "current_mean", "temperature_mean", "discharge_capacity", "c_rate"]
            features = [v_mean, v_std, i_mean, temp, d_cap, c_rate]
            sequence.append(features)
        
        return np.array(sequence)


# =============================================================================
# PART 3: INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_resources()

    def _load_resources(self):
        """Load trained model and scaler."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

        # Load Scaler
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        
        # Load Model
        self.model = ChargeGPTModel()
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"✅ Inference Engine initialized using model from epoch {checkpoint.get('epoch', '?')}")

    def predict_future_soh(self, c_rate: float, current_soh: float = 1.0):
        """
        Predicts the SOH after a sequence of charging at the given C-rate.
        
        Args:
            c_rate (float): Charging speed (e.g., 0.5, 1.0, 3.0).
            current_soh (float): Current State of Health (0.0 to 1.0).
            
        Returns:
            dict: {
                "predicted_soh": float,
                "degradation_impact": float,
                "physics_features": dict
            }
        """
        # 1. Generate synthetic features based on physics
        raw_sequence = PhysicsSimulator.generate_sequence(c_rate, current_soh)
        
        # 2. Normalize features using the loaded scaler
        # Scaler expects (n_samples, n_features). We have (seq_len, n_features).
        scaled_sequence = self.scaler.transform(raw_sequence)
        
        # 3. Convert to Tensor
        # Model expects (batch_size, seq_len, n_features)
        input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(device)
        
        # 4. Predict
        with torch.no_grad():
            predicted_soh = self.model(input_tensor).item()
            
        # 5. Calculate impact
        # Degradation = (Current SOH - Predicted SOH)
        # However, the model predicts the *absolute* SOH.
        # If the model predicts a HIGHER SOH than current, clap it (entropy only increases).
        if predicted_soh > current_soh:
            predicted_soh = current_soh
            
        degradation = current_soh - predicted_soh
        
        return {
            "predicted_soh": round(predicted_soh, 4),
            "degradation_impact": round(degradation, 6),
            "input_c_rate": c_rate,
            "simulated_temp": raw_sequence[0][3] # temperature is at index 3
        }

# Simple test if run directly
if __name__ == "__main__":
    engine = InferenceEngine()
    
    print("\n--- Running Simulation Scenarios ---")
    scenarios = [0.5, 1.0, 3.0]
    
    for rate in scenarios:
        result = engine.predict_future_soh(c_rate=rate, current_soh=0.95)
        print(f"Scenario: Charge at {rate}C")
        print(f"  -> Simulated Temp: {result['simulated_temp']}°C")
        print(f"  -> Predicted SOH:  {result['predicted_soh']*100:.2f}%")
        print(f"  -> Impact:         -{result['degradation_impact']*100:.4f}% health")
        print()

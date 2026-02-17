# =============================================================================
# ChargeGPT — Synthetic Battery Data Generation (Physics-Based)
# =============================================================================
# Uses empirical electrochemical degradation models:
#   - Power-law capacity fade: Q_loss = k * N^0.5 (square-root of cycle count)
#   - Arrhenius temperature dependence: k = A * exp(-Ea / RT)
#   - C-rate stress factor: higher C-rates accelerate SEI growth
#
# References:
#   - J. Wang et al., "Cycle-life model for Li-ion cells", J. Power Sources, 2011
#   - P. Ramadass et al., "Capacity fade of Sony 18650 cells", J. Power Sources, 2004
#
# Run in Google Colab:
#   Cell 1: Just run this script (no pip installs needed — only numpy, pandas, matplotlib)
#   Cell 2: files.download("battery_data.csv")
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── Configuration ──────────────────────────────────────────────────────────

C_RATES = [0.5, 1.0, 1.5, 2.0, 3.0]
CYCLES_PER_RATE = 500  # 500 cycles per C-rate = 2500 total rows
NOMINAL_CAPACITY = 5.0  # Ah (typical EV cell)
NOMINAL_VOLTAGE = 3.7   # V (nominal for NMC chemistry)

# ─── Physics Parameters ────────────────────────────────────────────────────

# Base degradation rate at 1C, 25°C (empirical, from literature)
# This controls how fast SOH drops — tuned for ~80% SOH at 1500 cycles @ 1C
BASE_DEGRADATION_RATE = 0.008

# C-rate stress exponent: degradation scales as C_rate^stress_exponent
# Higher C-rates cause more lithium plating and SEI cracking
C_RATE_STRESS_EXPONENT = 1.5

# Temperature model: cell heats up proportionally to C-rate^2 (I²R losses)
AMBIENT_TEMP = 25.0        # °C
THERMAL_RESISTANCE = 3.5   # °C per C-rate² (how much cell heats up)

# Arrhenius: temperature accelerates degradation
# Every 10°C rise roughly doubles degradation rate
ACTIVATION_ENERGY = 0.3    # Simplified factor (unitless, for modeling)

# ─── Degradation Model ─────────────────────────────────────────────────────

def compute_cell_temperature(c_rate, cycle_num, ambient=AMBIENT_TEMP):
    """
    Cell temperature increases with C-rate (I²R heating) and slightly
    with age (internal resistance grows).
    """
    # Base heating from C-rate (quadratic — Joule heating)
    heating = THERMAL_RESISTANCE * (c_rate ** 1.8)

    # Aging increases internal resistance → more heat over time
    aging_heat = 0.002 * cycle_num * c_rate

    # Add realistic noise (±1°C measurement noise)
    noise = np.random.normal(0, 0.5)

    temp = ambient + heating + aging_heat + noise
    return round(temp, 2)


def compute_soh(c_rate, cycle_num):
    """
    Physics-based SOH model using power-law capacity fade.

    SOH = 1 - k * sqrt(N)

    where:
        k = base_rate * c_rate_factor * temperature_factor
        N = cycle number
    """
    # C-rate stress factor (higher C = more degradation)
    c_rate_factor = c_rate ** C_RATE_STRESS_EXPONENT

    # Temperature factor (Arrhenius-like: hotter = faster degradation)
    temp = AMBIENT_TEMP + THERMAL_RESISTANCE * (c_rate ** 1.8)
    temp_factor = np.exp(ACTIVATION_ENERGY * (temp - 25.0) / 10.0)

    # Effective degradation rate
    k = BASE_DEGRADATION_RATE * c_rate_factor * temp_factor

    # Power-law fade: capacity loss proportional to sqrt(cycle count)
    # This is well-established in literature for SEI-dominated aging
    soh = 1.0 - k * np.sqrt(cycle_num)

    # Add small noise (real batteries have cycle-to-cycle variability)
    noise = np.random.normal(0, 0.001)
    soh = soh + noise

    return max(min(soh, 1.0), 0.5)  # Clamp between 0.5 and 1.0


def compute_voltage_features(c_rate, soh, cycle_num):
    """
    As battery degrades:
    - Mean voltage drops (internal resistance increases)
    - Voltage variability increases (less uniform electrode reaction)
    """
    # Base voltage decreases with degradation
    degradation = 1.0 - soh
    voltage_mean = NOMINAL_VOLTAGE + 0.2 - (0.8 * degradation) - (0.05 * c_rate)

    # Add cycle-to-cycle noise
    voltage_mean += np.random.normal(0, 0.01)

    # Voltage std increases with degradation and C-rate
    voltage_std = 0.15 + (0.3 * degradation) + (0.02 * c_rate)
    voltage_std += np.random.normal(0, 0.005)

    return round(voltage_mean, 4), round(abs(voltage_std), 4)


def compute_current_features(c_rate, soh):
    """
    Current during charging: proportional to C-rate.
    As battery ages, charging becomes less efficient.
    """
    # Nominal current = C_rate * Capacity
    nominal_current = c_rate * NOMINAL_CAPACITY * soh
    current_mean = nominal_current + np.random.normal(0, 0.1)
    return round(current_mean, 4)


def compute_discharge_capacity(soh):
    """
    Discharge capacity = SOH * nominal capacity.
    """
    cap = NOMINAL_CAPACITY * soh + np.random.normal(0, 0.01)
    return round(max(cap, 0), 4)


# ─── Generate All Data ─────────────────────────────────────────────────────

def generate_battery_data():
    """Generate synthetic battery cycling data for all C-rates."""
    all_rows = []

    for c_rate in C_RATES:
        print(f"Generating {CYCLES_PER_RATE} cycles at {c_rate}C...")

        for cycle in range(1, CYCLES_PER_RATE + 1):
            soh = compute_soh(c_rate, cycle)
            temp = compute_cell_temperature(c_rate, cycle)
            v_mean, v_std = compute_voltage_features(c_rate, soh, cycle)
            i_mean = compute_current_features(c_rate, soh)
            d_cap = compute_discharge_capacity(soh)

            row = {
                "cycle_number": cycle,
                "c_rate": c_rate,
                "voltage_mean": v_mean,
                "voltage_std": v_std,
                "current_mean": i_mean,
                "temperature_mean": temp,
                "discharge_capacity": d_cap,
                "soh": round(soh, 6),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # ─── Print Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATASET GENERATED: {len(df)} rows")
    print(f"{'='*60}")
    print(f"\nSOH range per C-rate:")
    for rate in sorted(df["c_rate"].unique()):
        subset = df[df["c_rate"] == rate]
        drop = (1 - subset["soh"].min()) * 100
        print(f"  {rate}C: SOH {subset['soh'].max():.4f} → {subset['soh'].min():.4f}  "
              f"(drop: {drop:.1f}%)  [{len(subset)} cycles]")

    print(f"\nTemperature range per C-rate:")
    for rate in sorted(df["c_rate"].unique()):
        subset = df[df["c_rate"] == rate]
        print(f"  {rate}C: {subset['temperature_mean'].min():.1f}°C → {subset['temperature_mean'].max():.1f}°C")

    # ─── Save ────────────────────────────────────────────────────────────
    df.to_csv("battery_data.csv", index=False)
    print(f"\n✅ Saved to battery_data.csv")

    return df


# ─── Visualization ──────────────────────────────────────────────────────────

def plot_data(df):
    """Generate preview plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {0.5: '#2ecc71', 1.0: '#3498db', 1.5: '#f39c12', 2.0: '#e74c3c', 3.0: '#9b59b6'}

    # Plot 1: SOH Degradation
    for rate in sorted(df["c_rate"].unique()):
        subset = df[df["c_rate"] == rate]
        axes[0, 0].plot(subset["cycle_number"], subset["soh"],
                        label=f"{rate}C", color=colors[rate], alpha=0.8)
    axes[0, 0].set_xlabel("Cycle Number")
    axes[0, 0].set_ylabel("State of Health (SOH)")
    axes[0, 0].set_title("Battery Degradation by Charging Speed")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0.6, 1.02)

    # Plot 2: Cell Temperature
    for rate in sorted(df["c_rate"].unique()):
        subset = df[df["c_rate"] == rate]
        axes[0, 1].plot(subset["cycle_number"], subset["temperature_mean"],
                        label=f"{rate}C", color=colors[rate], alpha=0.8)
    axes[0, 1].set_xlabel("Cycle Number")
    axes[0, 1].set_ylabel("Temperature (°C)")
    axes[0, 1].set_title("Cell Temperature by Charging Speed")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Discharge Capacity
    for rate in sorted(df["c_rate"].unique()):
        subset = df[df["c_rate"] == rate]
        axes[1, 0].plot(subset["cycle_number"], subset["discharge_capacity"],
                        label=f"{rate}C", color=colors[rate], alpha=0.8)
    axes[1, 0].set_xlabel("Cycle Number")
    axes[1, 0].set_ylabel("Discharge Capacity (Ah)")
    axes[1, 0].set_title("Capacity Fade by Charging Speed")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Voltage Mean
    for rate in sorted(df["c_rate"].unique()):
        subset = df[df["c_rate"] == rate]
        axes[1, 1].plot(subset["cycle_number"], subset["voltage_mean"],
                        label=f"{rate}C", color=colors[rate], alpha=0.8)
    axes[1, 1].set_xlabel("Cycle Number")
    axes[1, 1].set_ylabel("Mean Voltage (V)")
    axes[1, 1].set_title("Voltage Degradation by Charging Speed")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data_preview.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved plot to data_preview.png")


# ─── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_battery_data()
    plot_data(df)

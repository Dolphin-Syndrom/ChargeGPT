<div align="center">

# ⚡ ChargeGPT

### Autonomous Battery Health Optimization using Physics-Informed Natural Law Models

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

*A Physics-Informed Transformer that understands electrochemical degradation laws, wrapped with an autonomous AI agent that acts as your EV battery's guardian.*

</div>

---

## 🧠 What is ChargeGPT?

**ChargeGPT** is an advanced **Natural Law Model (NLM)** designed to extend the lifespan of Electric Vehicle (EV) batteries. Unlike traditional Battery Management Systems (BMS) that rely on simple rule-based logic, ChargeGPT utilizes a **Physics-Informed Transformer** to understand the underlying electrochemical laws of lithium-ion degradation.

By integrating this NLM with an **autonomous AI agent**, the system acts as a proactive **"Battery Guardian,"** negotiating optimal charging schedules that balance user needs with microscopic preservation of the battery's anode and cathode.

### 🔑 Key Innovation

> **Traditional BMS**: *"Battery is at 80%. Slow down charging."* (Rule-based)
>
> **ChargeGPT**: *"Based on electrochemical analysis, your current 2C fast-charging pattern will degrade the SEI layer by 18% over 500 cycles. Switching to 1C overnight charging would preserve 94% SOH — extending battery life by ~3 years."* (Physics-informed reasoning)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ChargeGPT System                       │
│                                                             │
│  ┌────────────────┐   ┌──────────────────┐   ┌───────────┐  │
│  │ Physics-Based  │──▶│   Transformer    │◀──│ LangChain │  │
│  │ Data Generator │   │   (NLM Brain)    │   │   Agent   │  │
│  │                │   │                  │   │           │  │
│  │ • Power-law    │   │ • 2-layer encoder│   │ • Gemini  │  │
│  │   capacity fade│   │ • 4 attn heads   │   │   LLM     │  │
│  │ • Arrhenius    │   │ • 76K params     │   │ • Predict │  │
│  │   temp effects │   │ • R² = 0.997     │   │   tool    │  │
│  │ • SEI growth   │   │                  │   │ • Compare │  │
│  │   modeling     │   │                  │   │   tool    │  │
│  └────────────────┘   └──────────────────┘   └───────────┘  │
│                              │                              │
│                              ▼                              │
│                    ┌──────────────────┐                      │
│                    │    Streamlit     │                      │
│                    │    Dashboard     │                      │
│                    │                  │                      │
│                    │ • Health Gauge   │                      │
│                    │ • Degradation    │                      │
│                    │   Charts         │                      │
│                    │ • Agent Chat     │                      │
│                    └──────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ How It Works

### Phase 1: Physics-Based Data Generation

We generate synthetic battery cycling data using **empirical electrochemical models** grounded in real physics:

| Model | Equation | What It Captures |
|-------|----------|-----------------|
| **Power-law capacity fade** | `SOH = 1 - k × √N` | SEI layer growth follows square-root of cycle count |
| **C-rate stress** | `k ∝ C_rate^1.5` | Higher charging speeds cause exponentially more damage |
| **Arrhenius temperature** | `k ∝ exp(Ea/RT)` | Every 10°C rise roughly doubles degradation rate |
| **Joule heating** | `ΔT ∝ C_rate^1.8` | Fast charging generates more internal heat |

**Dataset**: 2,500 cycles across 5 C-rates (0.5C → 3.0C), with 6 features per cycle.

<div align="center">

![Battery Degradation Data](models/data_preview.png)

*Synthetic degradation curves showing realistic physics: 3C charging degrades 8× faster than 0.5C*

</div>

### Phase 2: Transformer Model (The NLM)

A compact **Transformer Encoder** learns the degradation laws from data:

| Component | Specification |
|-----------|--------------|
| Architecture | Transformer Encoder |
| Input | Sliding window of 10 consecutive cycles (6 features each) |
| Encoder Layers | 2 |
| Attention Heads | 4 |
| Hidden Dimension | 64 |
| Activation | GELU |
| Output | SOH prediction ∈ [0, 1] via Sigmoid |
| Parameters | **76,033** |

**Why a Transformer?** The self-attention mechanism identifies *which past cycles* contributed most to current degradation — a single high-stress cycle causes disproportionate damage, and attention captures this naturally.

### Phase 3: Autonomous Agent

A **LangChain-powered AI agent** uses the trained Transformer as a tool:

1. User asks: *"Should I fast charge my EV daily?"*
2. Agent calls `predict_battery_health(c_rate=3.0, cycles=500)`
3. Transformer predicts: SOH = 61%
4. Agent compares with `predict_battery_health(c_rate=1.0, cycles=500)` → SOH = 82%
5. Agent responds with physics-backed recommendation

---

## 📊 Model Performance

<div align="center">

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | **0.88%** | Average prediction error less than 1% |
| **RMSE** | 0.98% | Root mean squared error |
| **R² Score** | **0.9971** | Explains 99.7% of variance |
| **MSE** | 0.0001 | Near-zero mean squared error |

![Model Predictions](models/prediction_results.png)

*Left: Predicted vs Actual SOH (tight alignment). Right: Error distribution centered near zero.*

![Training History](models/training_history.png)

*Training converges in ~45 epochs with LR scheduling and early stopping.*

</div>

---

## 📂 Project Structure

```
chargeGPT/
├── reference/
│   ├── generate_data_colab.py     # Physics-based data generation (Colab)
│   └── train_model_colab.py       # Transformer training pipeline (Colab)
├── data/
│   └── battery_data.csv           # 2,500 cycles of synthetic battery data
├── models/
│   ├── chargegpt_model.pth        # Trained Transformer weights
│   ├── scaler.pkl                 # Feature normalization scaler
│   └── model_config.json          # Architecture config + metrics
├── agent.py                       # LangChain AI agent (coming soon)
├── app.py                         # Streamlit dashboard (coming soon)
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Colab (for training) or local GPU
- Google Gemini API key (for the AI agent)

### Installation

```bash
git clone https://github.com/your-username/chargeGPT.git
cd chargeGPT
pip install -r requirements.txt
```

### Step 1: Generate Data (Colab)

Open `reference/generate_data_colab.py` in Google Colab and run it. Download `battery_data.csv` to `data/`.

### Step 2: Train Model (Colab)

Open `reference/train_model_colab.py` in Google Colab and run it. Download model files to `models/`.

### Step 3: Run the Agent

```bash
# Set your API key
export GOOGLE_API_KEY="your-gemini-key"

# Run the agent standalone
python agent.py
```

### Step 4: Launch Dashboard

```bash
streamlit run app.py
```

---

## 🔬 The Science Behind ChargeGPT

### Why Batteries Degrade

Lithium-ion batteries degrade through several mechanisms:

1. **SEI Layer Growth**: A protective film on the anode thickens over time, consuming active lithium and increasing resistance. Growth follows a `√t` law.

2. **Lithium Plating**: At high C-rates or low temperatures, lithium deposits as metallic film instead of intercalating — irreversible capacity loss.

3. **Mechanical Stress**: Volume changes during charge/discharge cause micro-cracks in electrode particles, exposing fresh surface area for SEI growth.

### How ChargeGPT Models This

| Degradation Mechanism | How We Capture It |
|----------------------|-------------------|
| SEI Growth | Power-law capacity fade: `Q_loss = k × √N` |
| C-rate Damage | Stress exponent: `k ∝ C^1.5` |
| Thermal Acceleration | Arrhenius: `k ∝ exp(Ea × ΔT / 10)` |
| Aging Resistance | Temperature rises with cycle count |

The Transformer learns these relationships implicitly from the data, discovering the *same laws* that electrochemists derived theoretically.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Generation | NumPy, Pandas | Physics-based synthetic battery data |
| Model | PyTorch | Transformer encoder for SOH prediction |
| Agent | LangChain + Gemini | Autonomous reasoning & tool use |
| Dashboard | Streamlit + Plotly | Interactive visualization |
| Preprocessing | scikit-learn | Feature normalization (MinMaxScaler) |

---

## 📚 References

- **Doyle, Fuller, Newman** (1993) — *Modeling of Galvanostatic Charge and Discharge of the Lithium/Polymer/Insertion Cell* — Foundation of electrochemical battery modeling
- **Wang et al.** (2011) — *Cycle-life model for graphite-LiFePO₄ cells* — Power-law capacity fade model
- **Ramadass et al.** (2004) — *Development of First Principles Capacity Fade Model for Li-Ion Cells* — SEI growth modeling
- **Severson et al.** (2019) — *Data-driven prediction of battery cycle life before capacity degradation* — ML for battery health

---

## 📝 License

This project is for educational and research purposes.

---

<div align="center">

**Built with ⚡ by ChargeGPT Team**

*Protecting EV batteries, one charge at a time.*

</div>

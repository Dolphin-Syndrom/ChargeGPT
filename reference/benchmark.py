import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("data/battery_data.csv")
features = ["voltage_mean", "voltage_std", "current_mean", "temperature_mean", "discharge_capacity", "c_rate"]
target = "soh"

X = df[features].values
y = df[target].values

# 2. Split Data (Using same random state as training for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model 1: Linear Regression (Baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 4. Model 2: Random Forest (Strong ML Baseline)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 5. Your Transformer Results (Hardcoded from your README for comparison)
# Replace these with actual values if you run inference
transformer_mae = 0.0088  # 0.88%
transformer_r2 = 0.9971

# 6. Evaluation
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:20} | MAE: {mae:.4f} ({mae*100:.2f}%) | R²: {r2:.4f}")
    return mae, r2

print("="*60)
print(" BENCHMARK RESULTS")
print("="*60)
evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Random Forest", y_test, y_pred_rf)
print(f"{'ChargeGPT (Ours)':20} | MAE: {transformer_mae:.4f} ({transformer_mae*100:.2f}%) | R²: {transformer_r2:.4f}")
print("="*60)

# 7. Plot Comparison
models = ['Linear Reg', 'Random Forest', 'ChargeGPT']
maes = [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf), transformer_mae]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, maes, color=['#95a5a6', '#34495e', '#2ecc71'])
plt.ylabel('Mean Absolute Error (Lower is Better)')
plt.title('Model Comparison: SOH Prediction Error')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.savefig("benchmark_comparison.png")
print("\n📊 Saved comparison plot to benchmark_comparison.png")
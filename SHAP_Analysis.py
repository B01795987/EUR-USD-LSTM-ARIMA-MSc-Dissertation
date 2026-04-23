
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION
CSV_PATH     = 'EURUSD_15min_Features.csv'
FEATURES     = ['Close', 'SMA_20', 'Upper_Band', 'Lower_Band', 'RSI']
LOOK_BACK    = 120
N_FEATURES   = len(FEATURES)
SHAP_SAMPLES = 50

# 1. LOAD DATA AND MODEL
df    = pd.read_csv(CSV_PATH, index_col='DateTime', parse_dates=True)
data  = df[FEATURES].values
model = load_model('eurusd_lstm_final.keras')
print(f"Loaded {len(data)} rows")


# 2. SCALE — training data only
train_end   = int(len(data) * 0.8)
scaler      = MinMaxScaler()
scaler.fit(data[:train_end])
scaled_data = scaler.transform(data)


# 3. BUILD SEQUENCES
def create_sequences(scaled, look_back):
    X = []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i])
    return np.array(X)

X_all  = create_sequences(scaled_data, LOOK_BACK)
X_flat = X_all.reshape(X_all.shape[0], -1)
print(f"Sequences: {X_all.shape} | Flattened: {X_flat.shape}")


# 4. WRAPPER — stays entirely in scaled (0-1) space
# SHAP must operate in a consistent single scale
def predict_fn(x_flat):
    x_reshaped = x_flat.reshape((-1, LOOK_BACK, N_FEATURES))
    preds = model.predict(x_reshaped, verbose=0)
    return preds.flatten()

# Confirm wrapper output is stable
test_out = predict_fn(X_flat[-5:])
print(f"\nWrapper output sample: {test_out}")
print(f"Range: {test_out.min():.4f} to {test_out.max():.4f}")


# 5. NORMALISE BACKGROUND AND TEST TO SAME SCALE
# Use StandardScaler on flattened sequences so SHAP
# perturbations stay in a meaningful range
from sklearn.preprocessing import StandardScaler

flat_scaler    = StandardScaler()
X_flat_norm    = flat_scaler.fit_transform(X_flat)

background     = shap.kmeans(X_flat_norm[:200], 10)
test_instances = X_flat_norm[200:200 + SHAP_SAMPLES]


# 6. WRAPPER FOR NORMALISED INPUT
def predict_fn_norm(x_flat_norm):
    # Reverse the flat normalisation
    x_flat_orig = flat_scaler.inverse_transform(x_flat_norm)
    x_reshaped  = x_flat_orig.reshape((-1, LOOK_BACK, N_FEATURES))
    preds       = model.predict(x_reshaped, verbose=0)
    return preds.flatten()


# 7. SHAP CALCULATION
print(f"\nCalculating SHAP values for {SHAP_SAMPLES} samples...")
explainer   = shap.KernelExplainer(predict_fn_norm, background)
shap_values = explainer.shap_values(test_instances, nsamples=100)

actual_shap = shap_values[0] if isinstance(shap_values, list) else shap_values
print(f"\nSHAP range: {actual_shap.min():.8f} to {actual_shap.max():.8f}")

if abs(actual_shap).max() > 1:
    print("WARNING: SHAP values still large — but proceeding with aggregation")
else:
    print("SHAP values look stable")


# 8. AGGREGATE ACROSS TIMESTEPS
shap_3d         = actual_shap.reshape(
    actual_shap.shape[0], LOOK_BACK, N_FEATURES
)
shap_aggregated = np.mean(np.abs(shap_3d), axis=1)
test_3d         = test_instances.reshape(-1, LOOK_BACK, N_FEATURES)
test_last       = test_3d[:, -1, :]


# 9. FEATURE IMPORTANCE RANKING
mean_shap = np.mean(shap_aggregated, axis=0)
fi        = pd.DataFrame({
    'Feature'    : FEATURES,
    'Mean |SHAP|': mean_shap
}).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)

print("\n" + "="*40)
print("FEATURE IMPORTANCE RANKING")
print("="*40)
print(fi.to_string(index=False))


# 10. PLOT 1 — Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_aggregated, test_last,
    feature_names=FEATURES,
    plot_type="dot",
    show=False, alpha=0.7
)
plt.title("XAI: Feature Importance Aggregated Across 120 Timesteps",
          fontsize=14, pad=20)
plt.xlabel("Mean |SHAP Value|", fontsize=11)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(axis='x', color='gray', linestyle='--', alpha=0.3)
cb_ax = plt.gcf().axes[-1]
cb_ax.set_ylabel('Feature Value (High=Red, Low=Blue)',
                 rotation=270, labelpad=20, fontsize=10)
plt.tight_layout()
plt.savefig('SHAP_Aggregated.png', dpi=300, bbox_inches='tight')
print("\nSaved: SHAP_Aggregated.png")


# 11. PLOT 2 — Temporal Heatmap
shap_time = np.mean(np.abs(shap_3d), axis=0)

plt.figure(figsize=(14, 5))
plt.imshow(shap_time.T, aspect='auto', cmap='YlOrRd')
plt.colorbar(label='Mean |SHAP|')
plt.yticks(range(N_FEATURES), FEATURES, fontsize=10)
plt.xlabel('Timestep (0 = 120 bars ago, 119 = most recent)', fontsize=11)
plt.title('Temporal SHAP Heatmap — Which Lags Drive Predictions?',
          fontsize=14)
plt.tight_layout()
plt.savefig('SHAP_Temporal_Heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: SHAP_Temporal_Heatmap.png")
print("\nSHAP analysis complete.")

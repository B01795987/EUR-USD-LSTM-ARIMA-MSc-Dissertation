import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION
CSV_PATH   = 'EURUSD_15min_Features.csv'
FEATURES   = ['Close', 'SMA_20', 'Upper_Band', 'Lower_Band', 'RSI']
LOOK_BACK  = 120        
N_SPLITS   = 5
EPOCHS     = 50         
BATCH_SIZE = 64       


# 1. LOAD DATA
df   = pd.read_csv(CSV_PATH, index_col='DateTime', parse_dates=True)
data = df[FEATURES].values
print(f"Loaded {len(data)} rows | {df.index[0]} → {df.index[-1]}")

# 2. HELPER FUNCTIONS
def create_sequences(scaled_data, look_back):
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)


def inverse_transform(preds, scaler, n_features):
    dummy       = np.zeros((len(preds), n_features))
    dummy[:, 0] = preds.flatten()
    return scaler.inverse_transform(dummy)[:, 0]


def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mean_squared_error'
    )
    return model


# 3. WALK-FORWARD VALIDATION
fold_size   = len(data) // (N_SPLITS + 1)
results     = []
all_preds   = []
all_actuals = []

print(f"\nStarting Walk-Forward Validation | {N_SPLITS} folds | fold size: {fold_size}")
print("="*60)

for fold in range(1, N_SPLITS + 1):
    train_end = fold * fold_size
    test_end  = train_end + fold_size

    train_data = data[:train_end]
    test_data  = data[train_end:test_end]

    fold_scaler  = MinMaxScaler()
    train_scaled = fold_scaler.fit_transform(train_data)
    test_scaled  = fold_scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled, LOOK_BACK)
    X_test,  y_test  = create_sequences(test_scaled,  LOOK_BACK)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,                  # increased from 3 — gives model more time
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,                  # halves learning rate when stuck
        patience=2,                  # acts faster than early stopping
        min_lr=1e-6,
        verbose=0
    )

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        shuffle=False,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0)

    y_pred   = inverse_transform(y_pred_scaled, fold_scaler, len(FEATURES))
    y_actual = inverse_transform(y_test,        fold_scaler, len(FEATURES))

    mae  = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    da   = np.mean(
        np.sign(np.diff(y_pred)) == np.sign(np.diff(y_actual))
    )

    results.append({'Fold': fold, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'DA': da})
    all_preds.extend(y_pred)
    all_actuals.extend(y_actual)

    print(f"Fold {fold} | MAE: {mae:.5f} | MSE: {mse:.5f} | RMSE: {rmse:.5f} | DA: {da:.3f}")


# 4. SUMMARY
df_results = pd.DataFrame(results)

print("\n" + "="*60)
print("LSTM WALK-FORWARD RESULTS SUMMARY")
print("="*60)
print(df_results.to_string(index=False))
print(f"\nMean | MAE:  {df_results['MAE'].mean():.5f}  ± {df_results['MAE'].std():.5f}")
print(f"Mean | MSE:  {df_results['MSE'].mean():.5f}  ± {df_results['MSE'].std():.5f}")
print(f"Mean | RMSE: {df_results['RMSE'].mean():.5f} ± {df_results['RMSE'].std():.5f}")
print(f"Mean | DA:   {df_results['DA'].mean():.3f}   ± {df_results['DA'].std():.3f}")


# 5. SAVE PREDICTIONS
np.savetxt('lstm_predictionsV1.csv', np.array(all_preds),   delimiter=',')
np.savetxt('lstm_actualsV1.csv',     np.array(all_actuals), delimiter=',')
print("\nSaved: lstm_predictionsV1.csv, lstm_actualsV1.csv")


# 6. RETRAIN FINAL MODEL FOR SHAP
print("\nRetraining final model")

train_end    = int(len(data) * 0.8)
final_scaler = MinMaxScaler()
train_scaled = final_scaler.fit_transform(data[:train_end])

X_train_full, y_train_full = create_sequences(train_scaled, LOOK_BACK)

final_model = build_model((LOOK_BACK, len(FEATURES)))
final_model.fit(
    X_train_full, y_train_full,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ],
    shuffle=False,
    verbose=1
)

final_model.save('eurusd_lstm_finalV1.keras')
print("Saved: eurusd_lstm_finalV1.keras")


# 7. PLOT — Predicted vs Actual (final fold)
plt.figure(figsize=(14, 5))
plt.plot(all_actuals[-fold_size:], label='Actual',    color='steelblue', linewidth=1)
plt.plot(all_preds[-fold_size:],   label='Predicted', color='coral',     linewidth=1, alpha=0.8)
plt.title('LSTM (Improved) — Predicted vs Actual Close Price (Final Fold)', fontsize=14)
plt.xlabel('Timestep', fontsize=11)
plt.ylabel('EUR/USD Price', fontsize=11)
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('LSTM_Predictions_ImprovedV1.png', dpi=300, bbox_inches='tight')
print("Saved: LSTM_Predictions_ImprovedV1.png")

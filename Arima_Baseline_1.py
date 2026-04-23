
import warnings
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# CONFIGURATION
CSV_PATH        = 'EURUSD_15min_Features.csv'
N_SPLITS        = 5
BEST_ORDER      = (0, 1, 2)
CHECKPOINT_FILE = 'arima_checkpoint.json'


# 1. LOADING DATA
df = pd.read_csv(CSV_PATH, index_col='DateTime', parse_dates=True)
df.dropna(subset=['Close'], inplace=True)
series = df['Close']
data   = series.values
print(f"Loaded {len(data)} rows | {df.index[0]} → {df.index[-1]}")


# 2. LOAD CHECKPOINT IF EXISTS
fold_size = len(data) // (N_SPLITS + 1)

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint = json.load(f)
    completed_folds = checkpoint['completed_folds']
    results         = checkpoint['results']
    all_preds       = checkpoint['all_preds']
    all_actuals     = checkpoint['all_actuals']
    print(f"\nCheckpoint found {max(completed_folds) + 1}")
    print(f"Completed folds so far: {completed_folds}")
else:
    completed_folds = []
    results         = []
    all_preds       = []
    all_actuals     = []
    print("\nNo checkpoint found")


# 3. WALK-FORWARD VALIDATION
print(f"\nARIMA Walk-Forward | Order: {BEST_ORDER} | {N_SPLITS} folds | fold size: {fold_size}")
print("="*60)

for fold in range(1, N_SPLITS + 1):

    # Skip already completed folds
    if fold in completed_folds:
        print(f"Fold {fold} already complete")
        continue

    train_end = fold * fold_size
    test_end  = train_end + fold_size

    train = series[:train_end]
    test  = series[train_end:test_end]

    history     = list(train)
    predictions = []

    print(f"\nFold {fold} — {len(test)} steps to forecast")

    for t in range(len(test)):

        if t % 500 == 0:
            print(f"  Fold {fold} | Step {t}/{len(test)}")

        try:
            fit  = ARIMA(history, order=BEST_ORDER).fit()
            yhat = fit.forecast(steps=1)[0]
        except:
            yhat = history[-1]

        predictions.append(yhat)
        history.append(test.iloc[t])

    predictions = np.array(predictions)
    actuals     = test.values

    mae  = mean_absolute_error(actuals, predictions)
    mse  = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    da   = np.mean(
        np.sign(np.diff(predictions)) == np.sign(np.diff(actuals))
    )

    naive     = np.array([history[i] for i in range(len(test))])
    naive_mae = mean_absolute_error(actuals, naive)
    naive_mse = mean_squared_error(actuals, naive)
    naive_da  = np.mean(
        np.sign(np.diff(naive)) == np.sign(np.diff(actuals))
    )

    results.append({
        'Fold'     : fold,
        'MAE'      : mae,
        'MSE'      : mse,
        'RMSE'     : rmse,
        'DA'       : da,
        'Naive_MAE': naive_mae,
        'Naive_MSE': naive_mse,
        'Naive_DA' : naive_da
    })

    all_preds.extend(predictions.tolist())
    all_actuals.extend(actuals.tolist())

    # Save checkpoint immediately after fold completes
    completed_folds.append(fold)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'completed_folds': completed_folds,
            'results'        : results,
            'all_preds'      : all_preds,
            'all_actuals'    : all_actuals
        }, f)

    print(f"Fold {fold} DONE | MAE: {mae:.5f} | MSE: {mse:.7f} | RMSE: {rmse:.5f} | DA: {da:.3f}")
    print(f"Checkpoint saved.")


# 4. SUMMARY
df_results = pd.DataFrame(results)

print("\n" + "="*60)
print("ARIMA WALK-FORWARD RESULTS")
print("="*60)
print(df_results[['Fold','MAE','MSE','RMSE','DA']].to_string(index=False))
print(f"\nARIMA  | MAE:  {df_results['MAE'].mean():.5f}  ± {df_results['MAE'].std():.5f}")
print(f"ARIMA  | MSE:  {df_results['MSE'].mean():.7f}  ± {df_results['MSE'].std():.7f}")
print(f"ARIMA  | RMSE: {df_results['RMSE'].mean():.5f} ± {df_results['RMSE'].std():.5f}")
print(f"ARIMA  | DA:   {df_results['DA'].mean():.3f}   ± {df_results['DA'].std():.3f}")
print(f"\nNaive  | MAE:  {df_results['Naive_MAE'].mean():.5f}")
print(f"Naive  | MSE:  {df_results['Naive_MSE'].mean():.7f}")
print(f"Naive  | DA:   {df_results['Naive_DA'].mean():.3f}")


# 5. SAVE PREDICTIONS
np.savetxt('arima_predictions.csv', np.array(all_preds),   delimiter=',')
np.savetxt('arima_actuals.csv',     np.array(all_actuals), delimiter=',')
df_results.to_csv('arima_results.csv', index=False)
print("\nSaved: arima_predictions.csv, arima_actuals.csv, arima_results.csv")


# 6. PLOT
final_actuals = np.array(all_actuals[-fold_size:])
final_preds   = np.array(all_preds[-fold_size:])
N = 200

fig, axes = plt.subplots(3, 1, figsize=(15, 18))

axes[0].plot(final_actuals[-N:], color='royalblue', label='Actual', linewidth=1.5)
axes[0].plot(final_preds[-N:],   color='crimson',   label='ARIMA Prediction',
             linestyle='--', linewidth=1.5)
axes[0].set_title(f'ARIMA {BEST_ORDER} Walk-Forward — Actual vs Predicted (Last {N} Bars)',
                  fontsize=14)
axes[0].set_xlabel('Timestep (15-Min Bars)')
axes[0].set_ylabel('EUR/USD Price')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

error = final_actuals[-N:] - final_preds[-N:]
axes[1].bar(range(len(error)), error,
            color=np.where(error >= 0, 'steelblue', 'salmon'), alpha=0.7)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('Prediction Error (Actual − Predicted)', fontsize=14)
axes[1].set_xlabel('Timestep (15-Min Bars)')
axes[1].set_ylabel('Error (USD)')
axes[1].grid(True, alpha=0.3)

actual_dir = np.sign(np.diff(final_actuals[-N:]))
pred_dir   = np.sign(np.diff(final_preds[-N:]))
correct    = actual_dir == pred_dir
fold_da    = df_results['DA'].iloc[-1]

axes[2].bar(range(len(correct)), np.ones(len(correct)),
            color=np.where(correct, 'seagreen', 'crimson'), alpha=0.7)
axes[2].set_title(
    f'Directional Accuracy — Green=Correct Red=Wrong | Final Fold DA: {fold_da:.3f}',
    fontsize=14)
axes[2].set_xlabel('Timestep (15-Min Bars)')
axes[2].set_yticks([])
axes[2].grid(False)

plt.tight_layout()
plt.savefig('ARIMA_WalkForward.png', dpi=300, bbox_inches='tight')
print("Saved: ARIMA_WalkForward.png")

# Clean up checkpoint after successful completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Checkpoint file removed, run complete.")
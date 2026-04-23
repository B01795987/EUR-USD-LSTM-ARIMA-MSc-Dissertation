
# DIEBOLD-MARIANO TEST — LSTM vs ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# 1. LOAD PREDICTIONS
lstm_preds   = np.loadtxt('lstm_predictions.csv',  delimiter=',')
lstm_actuals = np.loadtxt('lstm_actuals.csv',       delimiter=',')
arima_preds  = np.loadtxt('arima_predictions.csv',  delimiter=',')
arima_actuals= np.loadtxt('arima_actuals.csv',      delimiter=',')

print(f"LSTM  predictions loaded  : {len(lstm_preds)} samples")
print(f"ARIMA predictions loaded  : {len(arima_preds)} samples")


# 2. ALIGN LENGTHS
# LSTM loses look_back rows per fold due to sequence creation
# ARIMA predicts every step — lengths will differ
min_len      = min(len(lstm_preds), len(arima_preds))
lstm_preds   = lstm_preds[-min_len:]
lstm_actuals = lstm_actuals[-min_len:]
arima_preds  = arima_preds[-min_len:]
arima_actuals= arima_actuals[-min_len:]

print(f"Aligned length            : {min_len} samples")

# Use LSTM actuals as reference
actuals = lstm_actuals


# 3. DIEBOLD-MARIANO TEST FUNCTION
def diebold_mariano_test(actuals, pred1, pred2, model1_name, model2_name):
    """
    Tests whether forecast accuracy between two models is significantly different.

    H0: No difference in forecast accuracy between model1 and model2
    H1: There is a significant difference

    Negative DM statistic → model1 is more accurate
    Positive DM statistic → model2 is more accurate
    p < 0.05             → difference is statistically significant
    """

    e1 = actuals - pred1   # model1 errors
    e2 = actuals - pred2   # model2 errors

    d  = e1**2 - e2**2     # loss differential (squared error)

    T      = len(d)
    mean_d = np.mean(d)
    var_d  = np.var(d, ddof=1)

    DM_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * stats.t.sf(np.abs(DM_stat), df=T - 1)

    print(f"\n{'='*60}")
    print(f"DIEBOLD-MARIANO TEST: {model1_name} vs {model2_name}")
    print(f"{'='*60}")
    print(f"DM Statistic  : {DM_stat:.4f}")
    print(f"P-Value       : {p_value:.6f}")
    print(f"Sample Size   : {T}")
    print(f"Mean Loss Diff: {mean_d:.8f}")

    print(f"\nInterpretation:")
    if p_value < 0.01:
        significance = "highly significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    elif p_value < 0.10:
        significance = "marginally significant (p < 0.10)"
    else:
        significance = "not significant (p >= 0.10)"

    if p_value < 0.05:
        if DM_stat < 0:
            print(f"  {model1_name} is {significance}ally more accurate than {model2_name}")
        else:
            print(f"  {model2_name} is {significance}ally more accurate than {model1_name}")
    else:
        print(f"  No significant difference between {model1_name} and {model2_name}")
        print(f"  Result is {significance}")

    return DM_stat, p_value


# 4. RUN DM TEST
dm_stat, p_value = diebold_mariano_test(
    actuals, lstm_preds, arima_preds, 'LSTM', 'ARIMA'
)


# 5. FULL METRICS COMPARISON TABLE
def get_metrics(actuals, preds):
    mae  = np.mean(np.abs(actuals - preds))
    mse  = np.mean((actuals - preds)**2)
    rmse = np.sqrt(mse)
    da   = np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(actuals)))
    return mae, mse, rmse, da

lstm_mae,  lstm_mse,  lstm_rmse,  lstm_da  = get_metrics(actuals, lstm_preds)
arima_mae, arima_mse, arima_rmse, arima_da = get_metrics(actuals, arima_preds)

print(f"\n{'='*60}")
print("FULL COMPARISON TABLE")
print(f"{'='*60}")
comparison = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'DA'],
    'LSTM'  : [f"{lstm_mae:.5f}",  f"{lstm_mse:.7f}",
               f"{lstm_rmse:.5f}", f"{lstm_da:.3f}"],
    'ARIMA' : [f"{arima_mae:.5f}", f"{arima_mse:.7f}",
               f"{arima_rmse:.5f}",f"{arima_da:.3f}"],
    'Winner': [
        'ARIMA' if arima_mae  < lstm_mae  else 'LSTM',
        'ARIMA' if arima_mse  < lstm_mse  else 'LSTM',
        'ARIMA' if arima_rmse < lstm_rmse else 'LSTM',
        'ARIMA' if arima_da   > lstm_da   else 'LSTM'
    ]
})
print(comparison.to_string(index=False))

print(f"\nDM Test Summary:")
print(f"  DM Statistic : {dm_stat:.4f}")
print(f"  P-Value      : {p_value:.6f}")
print(f"  Conclusion   : {'Statistically significant difference' if p_value < 0.05 else 'No statistically significant difference'}")


# 6. SAVE RESULTS
dm_summary = pd.DataFrame({
    'Test'        : ['Diebold-Mariano'],
    'DM_Statistic': [dm_stat],
    'P_Value'     : [p_value],
    'Significant' : [p_value < 0.05],
    'Winner'      : ['ARIMA' if (p_value < 0.05 and dm_stat > 0) else
                     'LSTM'  if (p_value < 0.05 and dm_stat < 0) else
                     'No significant difference']
})
dm_summary.to_csv('dm_test_results.csv', index=False)
print("\nSaved: dm_test_results.csv")


# 7. VISUALISATION
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: MAE / RMSE bar comparison
metrics      = ['MAE', 'RMSE']
lstm_scores  = [lstm_mae,  lstm_rmse]
arima_scores = [arima_mae, arima_rmse]
x = np.arange(len(metrics))
w = 0.35

axes[0,0].bar(x - w/2, lstm_scores,  w, label='LSTM',  color='steelblue', capsize=5)
axes[0,0].bar(x + w/2, arima_scores, w, label='ARIMA', color='coral',     capsize=5)
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(metrics, fontsize=12)
axes[0,0].set_title('MAE & RMSE Comparison', fontsize=13)
axes[0,0].set_ylabel('Error (USD)', fontsize=11)
axes[0,0].legend(fontsize=11)
axes[0,0].grid(axis='y', linestyle='--', alpha=0.4)

# Plot 2: DA comparison
axes[0,1].bar(['LSTM', 'ARIMA'], [lstm_da, arima_da],
              color=['steelblue', 'coral'], width=0.4)
axes[0,1].axhline(0.5, color='red', linestyle='--',
                  linewidth=1.5, label='Random baseline (0.500)')
axes[0,1].set_title('Directional Accuracy Comparison', fontsize=13)
axes[0,1].set_ylabel('Directional Accuracy', fontsize=11)
axes[0,1].set_ylim(0.45, 0.55)
axes[0,1].legend(fontsize=10)
axes[0,1].grid(axis='y', linestyle='--', alpha=0.4)

# Plot 3: Prediction errors over time
lstm_errors  = actuals - lstm_preds
arima_errors = actuals - arima_preds

axes[1,0].plot(lstm_errors[-500:],  color='steelblue',
               label='LSTM errors',  alpha=0.7, linewidth=0.8)
axes[1,0].plot(arima_errors[-500:], color='coral',
               label='ARIMA errors', alpha=0.7, linewidth=0.8)
axes[1,0].axhline(0, color='black', linewidth=0.8)
axes[1,0].set_title('Prediction Errors Over Time (Last 500 Steps)', fontsize=13)
axes[1,0].set_xlabel('Timestep', fontsize=11)
axes[1,0].set_ylabel('Error (USD)', fontsize=11)
axes[1,0].legend(fontsize=11)
axes[1,0].grid(axis='y', linestyle='--', alpha=0.4)

# Plot 4: DM test result annotation
axes[1,1].axis('off')
result_text = (
    f"DIEBOLD-MARIANO TEST RESULTS\n\n"
    f"DM Statistic:  {dm_stat:.4f}\n"
    f"P-Value:       {p_value:.6f}\n\n"
    f"{'STATISTICALLY SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'}\n\n"
    f"LSTM  MAE:  {lstm_mae:.5f}\n"
    f"ARIMA MAE:  {arima_mae:.5f}\n\n"
    f"LSTM  DA:   {lstm_da:.3f}\n"
    f"ARIMA DA:   {arima_da:.3f}\n\n"
    f"{'ARIMA significantly more accurate' if (p_value < 0.05 and dm_stat > 0) else 'LSTM significantly more accurate' if (p_value < 0.05 and dm_stat < 0) else 'No significant difference detected'}"
)
axes[1,1].text(0.1, 0.5, result_text,
               transform=axes[1,1].transAxes,
               fontsize=12, verticalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow',
                         alpha=0.8))

plt.suptitle('LSTM vs ARIMA — Diebold-Mariano Forecast Comparison',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('DM_Test_Results.png', dpi=300, bbox_inches='tight')
print("Saved: DM_Test_Results.png")

print("\n" + "="*60)
print("DIEBOLD-MARIANO TEST COMPLETE")
print("Next step: Run SHAP analysis")
print("="*60)

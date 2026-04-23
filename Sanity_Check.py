import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('EURUSD_15min_Features.csv', index_col='DateTime', parse_dates=True)
data = df[['Close', 'SMA_20', 'Upper_Band', 'Lower_Band', 'RSI']].values

train_end = int(len(data) * 0.8)
scaler = MinMaxScaler()
scaler.fit(data[:train_end])
scaled = scaler.transform(data)

X = []
for i in range(120, len(scaled)):
    X.append(scaled[i-120:i])
X = np.array(X)

model = load_model('eurusd_lstm_final.keras')
test_preds = model.predict(X[-10:], verbose=0)
print("Sample predictions:")
print(test_preds.flatten())
print(f"Min: {test_preds.min():.6f} | Max: {test_preds.max():.6f}")
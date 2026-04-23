import pandas as pd

# 1. Loading master file
df = pd.read_csv('EURUSD_15min_Master.csv', index_col='DateTime', parse_dates=True)

# 2. Calculating a Simple Moving Average (20 periods = 5 hours of data)
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# 3. Bollinger Bands (The Volatility Indicator)
std_dev = df['Close'].rolling(window=20).std()
df['Upper_Band'] = df['SMA_20'] + (std_dev * 2)
df['Lower_Band'] = df['SMA_20'] - (std_dev * 2)

# 4. Calculating the RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1+rs))

# 4. Calculating Percentage Returns (Crucial for LSTM scaling)
df['Returns'] = df['Close'].pct_change()

# 5. Droping rows with "NaN" 
df.dropna(inplace=True)

# 6. Saving the file
df.to_csv('EURUSD_15min_Features.csv')

print("Features Added! Your dataset is now ready for the LSTM.")
print(df[['Close', 'SMA_20', 'RSI', 'Returns']].head())
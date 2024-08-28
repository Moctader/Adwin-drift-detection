import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river.drift import ADWIN

# Fetch financial data from Yahoo Finance
ticker = 'AAPL'  # Example: Apple Inc.
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Prepare the data
data['Return'] = data['Close'].pct_change()
data = data.dropna()
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Return']  # Regression target: return

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, shuffle=False
)

# Apply linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Get running MSE from regression to compare performance
predictions = lr.predict(X_test)
mse_orig = np.cumsum((predictions - y_test) ** 2)
mse_orig = mse_orig / np.arange(1, 1 + len(mse_orig))

# Initialize ADWIN for drift detection
adwin = ADWIN()

# Set up DataFrame to record results
status = pd.DataFrame(columns=["index", "y_true", "y_pred", "drift_detected"])
rec_list = []

# Run ADWIN
for i in range(len(X_train), len(X)):
    x = X_test.iloc[[i - len(X_train)]]
    y_pred = lr.predict(x)[0]
    y_true = y_test.iloc[i - len(X_train)]
    adwin.update((y_true - y_pred) ** 2)
    status.loc[i] = [i, y_true, y_pred, adwin.drift_detected]
    # If drift is detected, examine the window and retrain
    if adwin.drift_detected:
        retrain_start = i - len(adwin.window) + 1
        retrain_end = i + 1
        rec_list.append([retrain_start, retrain_end])
        X_train_new = X_test.iloc[retrain_start - len(X_train):retrain_end - len(X_train)]
        y_train_new = y_test.iloc[retrain_start - len(X_train):retrain_end - len(X_train)]
        lr = LinearRegression()
        lr.fit(X_train_new, y_train_new)

# Calculate running MSE
status['original_mse'] = mse_orig
status['mse'] = np.cumsum((status.y_true - status.y_pred) ** 2)
status['mse'] = status['mse'] / np.cumsum(np.repeat(1, status.shape[0]))

# Replace NaN and Inf values with finite numbers
status['original_mse'] = np.nan_to_num(status['original_mse'], nan=0.0, posinf=0.0, neginf=0.0)
status['mse'] = np.nan_to_num(status['mse'], nan=0.0, posinf=0.0, neginf=0.0)

# Convert rec_list to DataFrame with explicit column names
rec_list = pd.DataFrame(rec_list, columns=["start", "end"])

# Plot results
plt.figure(figsize=(16, 9))
plt.plot("index", "original_mse", data=status, label="Original MSE")
plt.plot("index", "mse", data=status, label="Retrain MSE")
plt.grid(False, axis="x")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("ADWIN Results: MSE", fontsize=22)
plt.ylabel("Value", fontsize=22)
plt.xlabel("Index", fontsize=22)
ylims = [0, max(status['original_mse'].max(), status['mse'].max()) * 1.1]
plt.ylim(ylims)
plt.axvspan(12000, len(X), alpha=0.2, label="Drift Induction Window")
plt.vlines(
    x=status.loc[status["drift_detected"] == True]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Drift Detected",
    color="red",
)
rec_list["y_val"] = np.linspace(
    start=0.05 * (ylims[1] - ylims[0]) + ylims[0],
    stop=0.2 * ylims[1],
    num=len(rec_list),
)
plt.hlines(
    y=rec_list["y_val"],
    xmin=rec_list["start"],
    xmax=rec_list["end"],
    color="green",
    label="Retraining Windows",
)
plt.legend(loc='lower right', fontsize='x-large')
plt.show()
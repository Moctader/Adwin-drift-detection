'''

Paired Learners is a concept where two models are trained together to detect concept drift.
One model is responsible for detecting changes in the data distribution, while the other model
is responsible for detecting changes in the performance of the primary model.



'''


import yfinance as yf
import pandas as pd
import numpy as np
from river.drift import ADWIN
from river import linear_model, metrics
import matplotlib.pyplot as plt

# Step 1: Load AAPL stock data using yfinance
def load_aapl_data(period="1y", interval="1d"):
    aapl_data = yf.download('AAPL', period=period, interval=interval)
    return aapl_data

# Step 2: Initialize drift detectors and a simple linear regression model
adwin_data = ADWIN()
adwin_performance = ADWIN()
model = linear_model.LinearRegression()
metric = metrics.MAE()

# Step 3: Detect concept drift in the stock data using Paired Learners
def detect_concept_drift(data, column='Close'):
    drift_points_data = []
    drift_points_performance = []
    
    # Iterate through stock data to detect drift in the 'Close' prices
    for i, (price, next_price) in enumerate(zip(data[column], data[column][1:])):
        # Update ADWIN for data distribution
        adwin_data.update(price)
        if adwin_data.drift_detected:
            drift_points_data.append((i, price))
            print(f"Data drift detected at index {i}, Price: {price:.2f}")
        
        # Update the model and ADWIN for performance
        y_pred = model.predict_one({'price': price}) or 0
        model.learn_one({'price': price}, next_price)
        metric.update(next_price, y_pred)
        loss = metric.get()
        adwin_performance.update(loss)
        if adwin_performance.drift_detected:
            drift_points_performance.append((i, price))
            print(f"Performance drift detected at index {i}, Price: {price:.2f}")
    
    return drift_points_data, drift_points_performance

# Step 4: Load data
aapl_data = load_aapl_data()

# Step 5: Detect drifts
drift_points_data, drift_points_performance = detect_concept_drift(aapl_data)

# Step 6: Plot the stock price and mark the detected drifts
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.plot(aapl_data['Close'], label='AAPL Close Price', color='blue')  # Plot the close price in blue

# Plot data drift points
if drift_points_data:
    indices, prices = zip(*drift_points_data)
    plt.scatter(indices, prices, color='red', label='Data Drift Detected', marker='x')

# Plot performance drift points
if drift_points_performance:
    indices, prices = zip(*drift_points_performance)
    plt.scatter(indices, prices, color='green', label='Performance Drift Detected', marker='o')

# Add title and labels
plt.title('AAPL Stock Price with Detected Drifts (Paired Learners)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend()  # Show the legend
plt.grid(True)  # Add grid for better readability
plt.show()  # Display the plot
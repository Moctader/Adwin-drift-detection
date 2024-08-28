import yfinance as yf
import pandas as pd
import numpy as np
from river.drift import ADWIN
import matplotlib.pyplot as plt

# Step 1: Load AAPL stock data using yfinance
def load_aapl_data(period="1y", interval="1d"):
    aapl_data = yf.download('AAPL', period=period, interval=interval)
    return aapl_data

# Step 2: Initialize ADWIN detector
adwin = ADWIN()

# Step 3: Detect concept drift in the stock data
def detect_concept_drift(data, column='Close'):
    drift_points = []
    
    # Iterate through stock data to detect drift in the 'Close' prices
    for i, price in enumerate(data[column]):
        in_drift = adwin.update(price)  # Update ADWIN with each price point
        
        # Check if a drift is detected
        if adwin.drift_detected:
            drift_points.append((i, price))
            print(f"Drift detected at index {i}, Price: {price:.2f}")
    
    return drift_points

# Step 4: Load data
aapl_data = load_aapl_data()

# Step 5: Detect drifts
drift_points = detect_concept_drift(aapl_data)

# Step 6: Plot the stock price and mark the detected drifts
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.plot(aapl_data['Close'], label='AAPL Close Price', color='blue')  # Plot the close price in blue

# If there are drift points, plot them
if drift_points:
    # Unzip the drift points into two lists: indices and prices
    indices, prices = zip(*drift_points)
    plt.scatter(indices, prices, color='red', label='Detected Drifts', marker='x')  # Mark drifts with red 'x'

# Add title and labels
plt.title('AAPL Stock Price with Detected Drifts (ADWIN)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend()  # Show the legend
plt.grid(True)  # Add grid for better readability
plt.show()  # Display the plot
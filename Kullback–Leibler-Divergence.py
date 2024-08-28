import yfinance as yf
import pandas as pd
import numpy as np
from river.drift import ADWIN
import matplotlib.pyplot as plt
from scipy.stats import entropy, norm

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
    for i, (date, price) in enumerate(data[column].items()):
        in_drift = adwin.update(price)  # Update ADWIN with each price point
        
        # Check if a drift is detected
        if adwin.drift_detected:
            drift_points.append((i, date, price))
            print(f"Drift detected at index {i}, Date: {date}, Price: {price:.2f}")
    
    return drift_points

# Step 4: Load data
aapl_data = load_aapl_data()

# Step 5: Detect drifts
drift_points = detect_concept_drift(aapl_data)

# Step 6: Calculate KL Divergence
def calculate_kl_divergence(data, drift_points, column='Close'):
    if not drift_points:
        return None
    
    # Get the indices of the drift points
    drift_indices = [point[0] for point in drift_points]
    
    # Split the data into segments based on drift points
    segments = np.split(data[column].values, drift_indices)
    
    # Calculate the probability distributions of the segments
    distributions = [np.histogram(segment, bins=10, density=True)[0] for segment in segments]
    
    # Calculate KL Divergence between consecutive segments
    kl_divergences = []
    for i in range(1, len(distributions)):
        p = distributions[i-1]
        q = distributions[i]
        kl_divergence = entropy(p, q)
        kl_divergences.append(kl_divergence)
    
    return kl_divergences, segments

kl_divergences, segments = calculate_kl_divergence(aapl_data, drift_points)
if kl_divergences:
    for i, kl_div in enumerate(kl_divergences):
        print(f"KL Divergence between segment {i} and {i+1}: {kl_div:.4f}")

# Step 7: Plot the stock price and mark the detected drifts
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.plot(aapl_data['Close'], label='AAPL Close Price', color='blue')  # Plot the close price in blue

# If there are drift points, plot them
if drift_points:
    # Unzip the drift points into three lists: indices, dates, and prices
    indices, dates, prices = zip(*drift_points)
    plt.scatter(indices, prices, color='red', label='Detected Drifts', marker='x')  # Mark drifts with red 'x'
    
    # Annotate the drift points with the index and price
    for i, (index, date, price) in enumerate(drift_points):
        plt.annotate(f'Index: {index}\nPrice: {price:.2f}', (index, price), textcoords="offset points", xytext=(0,10), ha='center')

# Add title and labels
plt.title('AAPL Stock Price with Detected Drifts (ADWIN)', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend()  # Show the legend
plt.grid(True)  # Add grid for better readability
plt.show()  # Display the plot

# Step 8: Plot Gaussian distributions for each segment
plt.figure(figsize=(10, 5))
x = np.linspace(min(aapl_data['Close']), max(aapl_data['Close']), 1000)

for i, segment in enumerate(segments):
    mu, std = norm.fit(segment)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, label=f'Segment {i+1} (from {drift_points[i-1][1].date() if i > 0 else aapl_data.index[0].date()} to {drift_points[i][1].date() if i < len(drift_points) else aapl_data.index[-1].date()})')

# Add vertical lines to indicate drift points
for index, date, _ in drift_points:
    plt.axvline(x=aapl_data['Close'].iloc[index], color='red', linestyle='--')

plt.title('Gaussian Distributions of Segments with Detected Drifts', fontsize=14)
plt.xlabel('Close Price', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Segments')  # Show the legend with a title
plt.grid(True)  # Add grid for better readability
plt.show()  # Display the plot
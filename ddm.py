from river import drift
import numpy as np

# Initialize ADWIN drift detector
detector = drift.ADWIN()

# Simulate streaming data
for x in np.random.normal(size=100):
    detector.update(x)
    if detector.drift_detected:
        print("Change detected!")
    else:
        print("No change.")
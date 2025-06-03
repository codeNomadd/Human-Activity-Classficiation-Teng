import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Directory containing the Excel files
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# File-label mapping
file_label_map = {
    'Walking.xlsx': 0,
    'Walking2.xlsx': 0,
    'Walking3.xlsx': 0,
    'Running.xlsx': 1,
    'Running2.xlsx': 1,
    'Running3.xlsx': 1,
    'Jumping.xlsx': 2,
    'Jumping2.xlsx': 2,
    'Jumping3.xlsx': 2,
}

SEGMENT_LENGTH = 1000
SEGMENTS_PER_FILE = 5
X = []
y = []

def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

for fname, label in file_label_map.items():
    fpath = os.path.join(DATA_DIR, fname)
    df = pd.read_excel(fpath)
    # Assume Timestamp is first column, Voltage is second column
    timestamp = df.iloc[:, 0].values
    voltage = df.iloc[:, 1].values
    # Trim voltage to match timestamp length if needed
    if len(voltage) > len(timestamp):
        voltage = voltage[:len(timestamp)]
    # Split into 5 non-overlapping 1000-point segments
    for i in range(SEGMENTS_PER_FILE):
        start = i * SEGMENT_LENGTH
        end = start + SEGMENT_LENGTH
        if end <= len(voltage):
            segment = voltage[start:end]
            segment = normalize(segment)
            X.append(segment)
            y.append(label)

X = np.stack(X)  # shape (45, 1000)
y = np.array(y)  # shape (45,)

# Save as .npy files for easy loading later
np.save(os.path.join(DATA_DIR, 'X.npy'), X)
np.save(os.path.join(DATA_DIR, 'y.npy'), y)

print(f"Saved X shape: {X.shape}, y shape: {y.shape} in {DATA_DIR}") 
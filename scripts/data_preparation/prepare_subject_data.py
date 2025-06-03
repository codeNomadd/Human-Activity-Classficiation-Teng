import os
import numpy as np
import pandas as pd

# Project root and data directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# File mapping for each person
person_files = {
    'P1': {'Walking': 'Walking.xlsx', 'Running': 'Running.xlsx', 'Jumping': 'Jumping.xlsx'},
    'P2': {'Walking': 'Walking2.xlsx', 'Running': 'Running2.xlsx', 'Jumping': 'Jumping2.xlsx'},
    'P3': {'Walking': 'Walking3.xlsx', 'Running': 'Running3.xlsx', 'Jumping': 'Jumping3.xlsx'},
}
activity_labels = {'Walking': 0, 'Running': 1, 'Jumping': 2}
SEGMENT_LENGTH = 500

def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

subject_data = {}

for person, files in person_files.items():
    X_segments = []
    y_labels = []
    for activity, fname in files.items():
        df = pd.read_excel(os.path.join(DATA_DIR, fname))
        voltage = df.iloc[:, 1].values
        # Slice into non-overlapping 500-point segments
        n_segments = len(voltage) // SEGMENT_LENGTH
        for i in range(n_segments):
            segment = voltage[i*SEGMENT_LENGTH:(i+1)*SEGMENT_LENGTH]
            segment = normalize(segment)
            X_segments.append(segment)
            y_labels.append(activity_labels[activity])
    subject_data[person] = {
        'X': np.stack(X_segments),
        'y': np.array(y_labels)
    }

# Save the dictionary as a .npz file for easy loading
np.savez(os.path.join(DATA_DIR, 'subject_data_500.npz'), **subject_data)
print("Saved subject_data_500.npz with keys:", list(subject_data.keys())) 
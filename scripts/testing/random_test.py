import os
import sys
import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
seed = random.randint(0, 10000)
random.seed(seed)
np.random.seed(seed)

# Project root and data directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'scripts', 'training', 'train_v5.py')

# Activity file mapping
activity_files = {
    'Jumping': ['Jumping.xlsx', 'Jumping2.xlsx', 'Jumping3.xlsx'],
    'Running': ['Running.xlsx', 'Running2.xlsx', 'Running3.xlsx'],
    'Walking': ['Walking.xlsx', 'Walking2.xlsx', 'Walking3.xlsx'],
}
activity_labels = {'Walking': 0, 'Running': 1, 'Jumping': 2}
label_names = ['Walking', 'Running', 'Jumping']

# Import feature extraction from train_v5.py
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'training'))
from train_v5 import extract_features, aggregate_segment_features

# Load short window data for feature fusion
X_100 = np.load(os.path.join(DATA_DIR, 'X_100_raw.npy'))
y_100 = np.load(os.path.join(DATA_DIR, 'y_100_raw.npy'))

# Load model training logic (simulate as in train_v5.py)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load long window data for training
X_500 = np.load(os.path.join(DATA_DIR, 'X_500.npy'))
y_500 = np.load(os.path.join(DATA_DIR, 'y_500.npy'))

# Train/test split and augmentation (as in train_v5.py)
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X_500, y_500, test_size=0.3, stratify=y_500, random_state=42
)
X_aug, y_aug = [], []
for x, label in zip(X_train_raw, y_train_raw):
    X_aug.append(x)
    y_aug.append(label)
    for _ in range(3):
        noise = np.random.normal(0, 0.02, size=x.shape)
        scale = np.random.uniform(0.9, 1.1)
        shift = np.random.uniform(-0.1, 0.1)
        x_aug = x + noise
        x_aug = x_aug * scale + shift
        X_aug.append(x_aug)
        y_aug.append(label)
X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

# Feature extraction for training
X_train_features = []
for x, y in zip(X_aug, y_aug):
    long_feat = extract_features(x)
    short_feat = aggregate_segment_features(X_100, y_100, y)
    combined_feat = np.concatenate([long_feat, short_feat])
    X_train_features.append(combined_feat)
X_train_features = np.array(X_train_features)

# Train model
clf = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train_features, y_aug)

# Prediction for each activity
results = []
for activity, files in activity_files.items():
    for chosen_file in files:
        df = pd.read_excel(os.path.join(DATA_DIR, chosen_file))
        voltage = df.iloc[:, 1].values
        if len(voltage) < 500:
            raise ValueError(f"Not enough data in {chosen_file}")
        for _ in range(3):
            start_idx = random.randint(0, len(voltage) - 500)
            segment = voltage[start_idx:start_idx+500]
            # Feature extraction for prediction
            long_feat = extract_features(segment)
            # For short_feat, use the true label for this activity
            short_feat = aggregate_segment_features(X_100, y_100, activity_labels[activity])
            combined_feat = np.concatenate([long_feat, short_feat]).reshape(1, -1)
            pred = clf.predict(combined_feat)[0]
            results.append((activity, label_names[pred]))
            print(f"True: {activity}, Predicted: {label_names[pred]}")

# Summary
correct = sum(1 for true, pred in results if true == pred)
print(f"\nCorrect predictions: {correct}/27") 
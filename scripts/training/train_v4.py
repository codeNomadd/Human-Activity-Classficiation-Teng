import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# === Feature Extraction Function === #
def extract_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['energy'] = np.sum(signal ** 2)
    features['skew'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)

    peaks, _ = find_peaks(signal)
    features['peak_count'] = len(peaks)
    features['peak_mean'] = np.mean(signal[peaks]) if len(peaks) > 0 else 0

    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['zero_crossings'] = len(zero_crossings)

    fft_vals = np.abs(fft(signal))[:len(signal) // 2]
    fft_energy = np.sum(fft_vals ** 2)
    features['fft_energy'] = fft_energy
    features['fft_low_ratio'] = np.sum(fft_vals[:100]) / fft_energy if fft_energy > 0 else 0
    features['fft_high_ratio'] = np.sum(fft_vals[100:]) / fft_energy if fft_energy > 0 else 0

    diff = np.diff(signal)
    features['slope_mean'] = np.mean(diff)
    features['slope_std'] = np.std(diff)
    features['slope_max'] = np.max(diff)
    features['slope_min'] = np.min(diff)

    return list(features.values())

# === Augmentation Function === #
def augment_signal(signal, jitter_std=0.02, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    signal_aug = signal.copy()
    signal_aug += np.random.normal(0, jitter_std, size=signal.shape)
    signal_aug *= np.random.uniform(*scale_range)
    signal_aug += np.random.uniform(*shift_range)
    return signal_aug

# === Load Dataset === #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
X_raw = np.load(os.path.join(DATA_DIR, 'X_500.npy'))  # shape (90, 500)
y_raw = np.load(os.path.join(DATA_DIR, 'y_500.npy'))  # shape (90,)

# === Split Before Augmentation === #
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

# === Apply Augmentations to Training === #
X_aug, y_aug = [], []
for x, label in zip(X_train_raw, y_train_raw):
    X_aug.append(x)
    y_aug.append(label)
    for _ in range(3):
        X_aug.append(augment_signal(x))
        y_aug.append(label)

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

# === Extract Features === #
X_train_feat = np.array([extract_features(x) for x in X_aug])
X_test_feat = np.array([extract_features(x) for x in X_test_raw])

# === Feature Selection === #
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_train_feat, y_aug)
X_train_sel = selector.transform(X_train_feat)
X_test_sel = selector.transform(X_test_feat)

# === Train Model === #
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_sel, y_aug)

# === Evaluate === #
y_pred = clf.predict(X_test_sel)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Walking", "Running", "Jumping"]))

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.fft import fft

# === Feature Extraction Functions === #
def extract_features(signal):
    features = {}
    
    # Basic stats
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['energy'] = np.sum(signal ** 2)
    
    # Peak info
    peaks, _ = find_peaks(signal)
    features['peak_count'] = len(peaks)
    features['peak_mean'] = np.mean(signal[peaks]) if len(peaks) > 0 else 0
    
    # Zero crossings
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['zero_crossings'] = len(zero_crossings)

    # FFT-based
    fft_vals = np.abs(fft(signal))[:len(signal)//2]
    fft_energy = np.sum(fft_vals ** 2)
    fft_band_low = np.sum(fft_vals[:100])
    fft_band_high = np.sum(fft_vals[100:])
    features['fft_energy'] = fft_energy
    features['fft_low'] = fft_band_low / fft_energy if fft_energy > 0 else 0
    features['fft_high'] = fft_band_high / fft_energy if fft_energy > 0 else 0

    return list(features.values())

# === Augmentation Function === #
def augment_signal(signal, jitter_std=0.02, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    signal_aug = signal.copy()
    signal_aug += np.random.normal(0, jitter_std, size=signal.shape)  # jitter
    signal_aug *= np.random.uniform(*scale_range)  # scale
    signal_aug += np.random.uniform(*shift_range)  # shift
    return signal_aug

# === Load Data === #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
X_raw = np.load(os.path.join(DATA_DIR, 'X.npy'))  # (45, 1000)
y_raw = np.load(os.path.join(DATA_DIR, 'y.npy'))

# === Train/Test Split Before Augmentation === #
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

# === Apply Augmentations to Training Set Only === #
X_augmented = []
y_augmented = []
for x, label in zip(X_train_raw, y_train_raw):
    X_augmented.append(x)
    y_augmented.append(label)
    for _ in range(3):  # generate 3 augmentations per sample
        x_aug = augment_signal(x)
        X_augmented.append(x_aug)
        y_augmented.append(label)

X_augmented = np.array(X_augmented)
y_augmented = np.array(y_augmented)

# === Extract Features === #
X_train_feat = np.array([extract_features(x) for x in X_augmented])
X_test_feat = np.array([extract_features(x) for x in X_test_raw])

# === Train Model === #
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_feat, y_augmented)

# === Evaluate === #
y_pred = clf.predict(X_test_feat)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Walking", "Running", "Jumping"]))

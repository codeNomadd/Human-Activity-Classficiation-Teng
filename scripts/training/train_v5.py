import os
import sys
import numpy as np
import random
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# === Handle seed argument ===
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 42
np.random.seed(seed)
random.seed(seed)

# === Project root and data directory ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# === Feature Extraction ===
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

# === Augmentation ===
def augment_signal(signal, jitter_std=0.02, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    signal_aug = signal.copy()
    signal_aug += np.random.normal(0, jitter_std, size=signal.shape)
    signal_aug *= np.random.uniform(*scale_range)
    signal_aug += np.random.uniform(*shift_range)
    return signal_aug

# === Load data ===
X_500 = np.load(os.path.join(DATA_DIR, 'X_500.npy'))  # (90, 500)
y_500 = np.load(os.path.join(DATA_DIR, 'y_500.npy'))  # (90,)
X_100 = np.load(os.path.join(DATA_DIR, 'X_100_raw.npy'))  # (450, 100)
y_100 = np.load(os.path.join(DATA_DIR, 'y_100_raw.npy'))  # (450,)

# === Split long samples ===
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X_500, y_500, test_size=0.3, stratify=y_500, random_state=seed
)

# === Augment long samples ===
X_aug, y_aug = [], []
for x, label in zip(X_train_raw, y_train_raw):
    X_aug.append(x)
    y_aug.append(label)
    for _ in range(3):
        X_aug.append(augment_signal(x))
        y_aug.append(label)
X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

# === Feature fusion: combine long sample features + short segment features ===
def aggregate_segment_features(X_short, y_short, label, method='mean'):
    """Return the aggregated feature vector (mean or max) for a class label."""
    seg_feats = np.array([extract_features(x) for x, y in zip(X_short, y_short) if y == label])
    return np.mean(seg_feats, axis=0) if method == 'mean' else np.max(seg_feats, axis=0)

X_train_features = []
for x, y in zip(X_aug, y_aug):
    long_feat = extract_features(x)
    short_feat = aggregate_segment_features(X_100, y_100, y)
    combined_feat = np.concatenate([long_feat, short_feat])
    X_train_features.append(combined_feat)
X_train_features = np.array(X_train_features)

X_test_features = []
for x, y in zip(X_test_raw, y_test):
    long_feat = extract_features(x)
    short_feat = aggregate_segment_features(X_100, y_100, y)
    combined_feat = np.concatenate([long_feat, short_feat])
    X_test_features.append(combined_feat)
X_test_features = np.array(X_test_features)

# === XGBoost Classifier ===
clf = XGBClassifier(n_estimators=200, random_state=seed, use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train_features, y_aug)

# === Evaluation ===
y_pred = clf.predict(X_test_features)
acc = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Walking", "Running", "Jumping"]))
print(f"Accuracy: {acc:.4f}")

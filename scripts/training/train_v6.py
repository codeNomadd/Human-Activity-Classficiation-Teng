import os
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Project root and data directory
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

def aggregate_segment_features(X_short, y_short, label, method='mean'):
    seg_feats = np.array([extract_features(x) for x, y in zip(X_short, y_short) if y == label])
    return np.mean(seg_feats, axis=0) if method == 'mean' else np.max(seg_feats, axis=0)

# Load subject-wise data
subject_data = np.load(os.path.join(DATA_DIR, 'subject_data_500.npz'), allow_pickle=True)
subjects = list(subject_data.keys())

# Load 100-point window data for feature fusion
X_100 = np.load(os.path.join(DATA_DIR, 'X_100_raw.npy'))
y_100 = np.load(os.path.join(DATA_DIR, 'y_100_raw.npy'))

results = []

for test_subject in subjects:
    # Prepare train and test sets
    X_test = subject_data[test_subject].item()['X']
    y_test = subject_data[test_subject].item()['y']
    X_train = []
    y_train = []
    for train_subject in subjects:
        if train_subject == test_subject:
            continue
        X_train.append(subject_data[train_subject].item()['X'])
        y_train.append(subject_data[train_subject].item()['y'])
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Feature extraction and fusion for training
    X_train_features = []
    for x, y in zip(X_train, y_train):
        long_feat = extract_features(x)
        short_feat = aggregate_segment_features(X_100, y_100, y)
        combined_feat = np.concatenate([long_feat, short_feat])
        X_train_features.append(combined_feat)
    X_train_features = np.array(X_train_features)

    # Feature extraction and fusion for testing
    X_test_features = []
    for x, y in zip(X_test, y_test):
        long_feat = extract_features(x)
        short_feat = aggregate_segment_features(X_100, y_100, y)
        combined_feat = np.concatenate([long_feat, short_feat])
        X_test_features.append(combined_feat)
    X_test_features = np.array(X_test_features)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_features, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test_features)
    acc = accuracy_score(y_test, y_pred)
    results.append(acc)
    print(f"\nLOSO Fold: Test Subject {test_subject}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Walking", "Running", "Jumping"]))

print(f"\nMean LOSO Accuracy: {np.mean(results):.4f}") 
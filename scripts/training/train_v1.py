import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
X = np.load(os.path.join(DATA_DIR, 'X.npy'))  # shape (45, 1000)
y = np.load(os.path.join(DATA_DIR, 'y.npy'))  # shape (45,)

# Augmentation function: jitter, scaling, shifting
def augment_batch(X, y, jitter_std=0.02, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    X_aug = []
    for x in X:
        aug_x = x.copy()

        # Jitter
        jitter = np.random.normal(0, jitter_std, size=aug_x.shape)
        aug_x += jitter

        # Scaling
        scale = np.random.uniform(*scale_range)
        aug_x *= scale

        # Shifting
        shift = np.random.uniform(*shift_range)
        aug_x += shift

        X_aug.append(aug_x)
    
    return np.array(X_aug), y.copy()

# Step 1: Original stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 2: Create 3 augmented versions of training data
X_jitter, y_jitter = augment_batch(X_train, y_train, jitter_std=0.03)
X_scale, y_scale = augment_batch(X_train, y_train, scale_range=(0.95, 1.05))
X_shift, y_shift = augment_batch(X_train, y_train, shift_range=(-0.05, 0.05))

# Step 3: Combine original and augmented training data
X_train_full = np.concatenate([X_train, X_jitter, X_scale, X_shift])
y_train_full = np.concatenate([y_train, y_jitter, y_scale, y_shift])

# Step 4: Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_full, y_train_full)

# Step 5: Predict and evaluate on clean test set
y_pred = clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Walking", "Running", "Jumping"]))

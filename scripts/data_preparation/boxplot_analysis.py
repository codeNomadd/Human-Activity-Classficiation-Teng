import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# -------------------------------
# Config
# -------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
X_path = os.path.join(DATA_DIR, 'X_100_raw.npy')
y_path = os.path.join(DATA_DIR, 'y_100_raw.npy')
CLASS_NAMES = ["Walking", "Running", "Jumping"]
WINDOW_SIZE = 100  # assuming total signal length is 500

# -------------------------------
# Load data
# -------------------------------
X = np.load(X_path)  # shape (90, 500)
y = np.load(y_path)  # shape (90,)

# Split train/test to focus on training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# Extract energy features per window
# -------------------------------
def extract_window_energies(signal, window_size=100):
    energies = []
    for i in range(0, len(signal), window_size):
        window = signal[i:i + window_size]
        if len(window) == window_size:
            energy = np.sum(window ** 2)  # total signal energy
            energies.append(energy)
    return energies  # list of 5 energy values

# -------------------------------
# Collect per-class energy values
# -------------------------------
window_energies = {cls: [] for cls in CLASS_NAMES}

for signal, label in zip(X_train, y_train):
    class_name = CLASS_NAMES[label]
    energies = extract_window_energies(signal)
    window_energies[class_name].extend(energies)

# -------------------------------
# Create DataFrame for plotting
# -------------------------------
import pandas as pd
df = []
for cls in CLASS_NAMES:
    for energy in window_energies[cls]:
        df.append({'Class': cls, 'WindowEnergy': energy})
df = pd.DataFrame(df)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='Class', y='WindowEnergy', data=df, palette="Set2")
plt.title("Windowed Total Signal Energy per Class")
plt.ylabel("Energy (Sum of Squared Voltage)")
plt.xlabel("Activity Class")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "window_energy_boxplot.png"))  # Save the figure
plt.show()

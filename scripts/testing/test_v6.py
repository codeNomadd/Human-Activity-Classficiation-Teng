import subprocess
import sys
import os
import re

# Path to the training script (robust, project-root relative)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train_script = os.path.join(PROJECT_ROOT, 'scripts', 'training', 'train_v6.py')

seeds = list(range(5))  # Run for 5 seeds
accuracies = []
summary_lines = []

for seed in seeds:
    print(f"Running train_v6.py with seed {seed}...")
    result = subprocess.run([sys.executable, train_script, str(seed)], capture_output=True, text=True)
    print(result.stdout)
    # Extract mean LOSO accuracy from output
    match = re.search(r"Mean LOSO Accuracy: ([0-9.]+)", result.stdout)
    if match:
        acc = float(match.group(1))
        accuracies.append(acc)
        summary_lines.append(f"Seed {seed}: Mean LOSO Accuracy = {acc}")
    else:
        summary_lines.append(f"Seed {seed}: Mean LOSO Accuracy not found")

summary_lines.append(f"\nMean of 5 runs: {sum(accuracies)/len(accuracies):.4f}" if accuracies else "No valid results.")

# Save summary to file
txt_path = os.path.join(os.path.dirname(__file__), 'test_v6_results.txt')
with open(txt_path, 'w') as f:
    for line in summary_lines:
        f.write(line + '\n')
print(f"\nResults summary saved to {txt_path}") 
import subprocess
import sys
import os

# Path to the training script (robust, project-root relative)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train_script = os.path.join(PROJECT_ROOT, 'scripts', 'training', 'train_v5.py')

seeds = list(range(5))  # Run for 5 seeds
accuracies = []

summary_lines = []

for seed in seeds:
    print(f"Running train_v5.py with seed {seed}...")
    result = subprocess.run([sys.executable, train_script, str(seed)], capture_output=True, text=True)
    print(result.stdout)
    # Extract accuracy from output (assuming train_v5.py prints 'Accuracy: <value>')
    for line in result.stdout.splitlines():
        if 'Accuracy:' in line:
            try:
                acc = float(line.split('Accuracy:')[1].strip().split()[0])
                accuracies.append(acc)
            except Exception:
                pass

summary_lines.append("\nSummary of 5 runs:")
for i, acc in enumerate(accuracies):
    line = f"Seed {seeds[i]}: Accuracy = {acc}"
    print(line)
    summary_lines.append(line)
if accuracies:
    mean_line = f"\nMean Accuracy: {sum(accuracies)/len(accuracies):.4f}"
    print(mean_line)
    summary_lines.append(mean_line)

# Save summary to file
txt_path = os.path.join(os.path.dirname(__file__), 'test_v5_results.txt')
with open(txt_path, 'w') as f:
    for line in summary_lines:
        f.write(line + '\n')
print(f"\nResults summary saved to {txt_path}") 
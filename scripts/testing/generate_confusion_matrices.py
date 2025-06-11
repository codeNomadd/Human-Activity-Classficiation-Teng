import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import traceback

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# Import training scripts
from scripts.training.train_v5 import train_and_evaluate as train_v5
from scripts.training.train_v6 import train_and_evaluate as train_v6

def plot_confusion_matrix(cm, labels, title, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    # Convert to integers if all values are whole numbers
    if np.all(np.mod(cm, 1) == 0):
        cm = cm.astype(int)
        fmt = 'd'
    else:
        fmt = '.2f'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(cm, labels, save_path):
    """Save confusion matrix to CSV."""
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(save_path)

def main():
    try:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Activity labels
        labels = ['Walking', 'Running', 'Jumping']

        # Train and evaluate v5
        print("Training and evaluating v5 model...")
        try:
            v5_cm, v5_acc = train_v5()
            
            # Save v5 results
            v5_cm_path = os.path.join(results_dir, 'confusion_matrix_v5.png')
            v5_csv_path = os.path.join(results_dir, 'confusion_matrix_v5.csv')
            plot_confusion_matrix(v5_cm, labels, 'Confusion Matrix - Model v5', v5_cm_path)
            save_confusion_matrix(v5_cm, labels, v5_csv_path)
            print(f"V5 Accuracy: {v5_acc:.4f}")
            print(f"V5 Confusion Matrix saved to {v5_cm_path}")
        except Exception as e:
            print(f"Error in v5 evaluation: {str(e)}")
            print("Traceback:")
            traceback.print_exc()

        # Train and evaluate v6
        print("\nTraining and evaluating v6 model...")
        try:
            v6_cm, v6_acc = train_v6()
            
            # Save v6 results
            v6_cm_path = os.path.join(results_dir, 'confusion_matrix_v6.png')
            v6_csv_path = os.path.join(results_dir, 'confusion_matrix_v6.csv')
            plot_confusion_matrix(v6_cm, labels, 'Confusion Matrix - Model v6', v6_cm_path)
            save_confusion_matrix(v6_cm, labels, v6_csv_path)
            print(f"V6 Accuracy: {v6_acc:.4f}")
            print(f"V6 Confusion Matrix saved to {v6_cm_path}")
        except Exception as e:
            print(f"Error in v6 evaluation: {str(e)}")
            print("Traceback:")
            traceback.print_exc()

        # Generate comparison report
        report_path = os.path.join(results_dir, 'model_comparison.txt')
        with open(report_path, 'w') as f:
            f.write("Model Comparison Report\n")
            f.write("=====================\n\n")
            f.write(f"Model v5 Accuracy: {v5_acc:.4f}\n")
            f.write(f"Model v6 Accuracy: {v6_acc:.4f}\n\n")
            f.write("V5 Confusion Matrix:\n")
            f.write(str(v5_cm))
            f.write("\n\nV6 Confusion Matrix:\n")
            f.write(str(v6_cm))
        
        print(f"\nComparison report saved to {report_path}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 
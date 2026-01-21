import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class WorkloadVisualizer:
    def __init__(self):
        # 1. Set general theme
        sns.set_theme(style="whitegrid")
        # 2. Set font globally for academic publication standards
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12
        })
        self.colors = {
            'truth': '#2c3e50', 
            'pred': '#e67e22', 
            'accent': '#3498db',
            'low_load': '#1f77b4', 
            'high_load': '#ff7f0e',
            'dist_bars': '#006666'
        }

    def calculate_stability_metrics(self, predictions):
        """Quantifies 'Label Flicker'. Higher score = More stable."""
        transitions = np.diff(predictions)
        flicker_count = np.count_nonzero(transitions)
        stability_index = 1 - (flicker_count / len(predictions))
        return stability_index

    # --- New: Signal Comparison (Recreates your Fig. 6) ---
    def plot_signal_comparison(self, time, low_signal, high_signal, modality="EEG"):
        """Plots signal morphology differences between load levels."""
        plt.figure(figsize=(8, 4))
        plt.plot(time, low_signal, label='low cognitive load', color=self.colors['low_load'])
        plt.plot(time, high_signal, label='high cognitive load', color=self.colors['high_load'])
        
        plt.title(f"Example of {modality} Signals in Different Scenarios")
        plt.xlabel("Timestamp (sec)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./result/comparison_{modality.lower()}.pdf")
        plt.close()

    # --- New: Distribution Grid (Recreates your Fig. 7) ---
    def plot_global_distributions(self, subject_data_dict):
        """Generates a grid of histograms for all 21 subjects."""
        n = len(subject_data_dict)
        cols = 7
        rows = (n // cols) + (1 if n % cols != 0 else 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, (sid, labels) in enumerate(subject_data_dict.items()):
            axes[i].hist(labels, bins=np.arange(1, 11)-0.5, color=self.colors['dist_bars'], rwidth=0.8)
            axes[i].set_title(f"({chr(97+i)}) Subject {sid}")
            axes[i].set_xticks(range(1, 10))
            axes[i].set_ylabel("Datapoints")
            axes[i].set_xlabel("Label")
            
        plt.suptitle("Self-reported Cognitive Load Level Distribution for Each Participant", y=1.02)
        plt.tight_layout()
        plt.savefig("./results/global_label_distribution.pdf")
        plt.close()

    def plot_results(self, ground_truth, predictions, subject_id):
        """Temporal Plot: Shows how the policy tracks workload over time."""
        stability = self.calculate_stability_metrics(predictions)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        time_seconds = np.arange(len(ground_truth)) * 2 
        
        ax.step(time_seconds, ground_truth, label='Ground Truth', 
                color=self.colors['truth'], alpha=0.4, where='post')
        ax.step(time_seconds, predictions, label=f'DRL Policy (SI: {stability:.2f})', 
                color=self.colors['pred'], where='post', linewidth=2)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Low', 'Med', 'High'])
        ax.set_xlabel('Time (Seconds)')
        ax.set_title(f'Temporal Stability: Subject {subject_id}')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"./results/temporal_subject_{subject_id}.pdf")
        plt.close()

    def plot_subject_confusion(self, y_true, y_pred, subject_id):
        """Individual Confusion Matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low', 'Med', 'High'],
                    yticklabels=['Low', 'Med', 'High'])
        plt.title(f"Confusion Matrix: Subject {subject_id}")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"./results/cm_subject_{subject_id}.pdf")
        plt.close()

    def plot_global_metrics(self, accuracies, stabilities):
        """Box Plot: Summary of LOSO performance across all subjects."""
        plt.figure(figsize=(7, 5))
        plt.boxplot([accuracies, stabilities], labels=['Accuracy', 'Stability Index'])
        plt.title("Overall Framework Performance (Across All Subjects)")
        plt.ylim(0, 1.05) 
        plt.savefig("./results/global_box_plot.pdf")
        plt.close()

    def plot_global_confusion_matrix(self, all_y_true, all_y_pred):
        """Normalized Global Confusion Matrix: Shows general error patterns."""
        cm = confusion_matrix(all_y_true, all_y_pred, normalize='true')
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap='Greens', fmt='.1%')
        plt.title('Normalized Global Confusion Matrix')
        plt.xticks([0.5, 1.5, 2.5], ['Low', 'Med', 'High'])
        plt.yticks([0.5, 1.5, 2.5], ['Low', 'Med', 'High'])
        plt.ylabel('Actual State')
        plt.xlabel('Predicted State')
        plt.savefig("./results/global_confusion_matrix.pdf")
        plt.close()

    def plot_performance_scatter(self, df_results):
        """Scatter Plot: Accuracy vs. Stability."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_results, x='Accuracy', y='Stability', 
                        s=100, color=self.colors['accent'])
        
        plt.axhline(df_results['Stability'].mean(), color='red', linestyle='--', alpha=0.5)
        plt.axvline(df_results['Accuracy'].mean(), color='red', linestyle='--', alpha=0.5)
        
        plt.title('Accuracy vs. Stability per Participant')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("./results/accuracy_stability_scatter.pdf")
        plt.close()
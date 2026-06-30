# SP-EEG-WorkloadPolicy

Official implementation of the paper: **"An Adaptive Policy-Based Framework for Driver Workload Sparse EEG Classification"** (Submitted to EMBC 2026).

## Why This Project?

### The Problem
Driver cognitive overload is a leading cause of road accidents. When a driver's mental workload exceeds safe limits, reaction times slow, situational awareness drops, and error rates climb. Real-time monitoring of cognitive workload could enable intelligent driver assistance systems to intervene before accidents happen.

### The Challenge
Existing approaches rely on high-density EEG caps (32-128 channels) that are expensive, uncomfortable, and impractical for in-vehicle use. Low-cost wearable EEG headbands like the **Muse S** (4 channels) are viable for real-world deployment, but their sparse signal makes accurate classification difficult. Moreover, traditional ML classifiers and deep learning models (CNNs) treat each time window independently, producing classifications that **flicker** between levels every 2 seconds — unusable in a real driving context where smooth, stable monitoring is essential.

### The Solution
This project reframes workload monitoring as a **sequential decision process** using Deep Reinforcement Learning (DRL). Instead of classifying each moment in isolation, a **Proximal Policy Optimization (PPO)** agent learns a policy that balances two objectives simultaneously:

1. **Accuracy** — Correctly identifying Low, Medium, and High workload states
2. **Stability** — Minimizing rapid transitions between states (label flickering)

The key innovation is a **Temporal Switching Penalty** embedded in the reward function: each time the agent changes its classification, it receives a penalty (-0.3). This teaches the policy to maintain stable outputs unless the EEG evidence truly warrants a transition — exactly what a real in-vehicle BCI needs.

### Key Results
- **Statistically significant** improvement over chance level (33%): p = 0.0307, t = 2.33
- Subjects achieve up to **100% accuracy** with perfect stability
- Mean accuracy: **54.8%** across 21 subjects (LOSO cross-validation)
- Mean stability index: **83.3%** — dramatically reducing label flickering
- Outperforms traditional ML baselines on stability-critical metrics

## Key Features

- **Policy-Based Classification:** Treats workload monitoring as a sequential decision process rather than isolated frame classification.
- **Stability Optimization:** Custom **Temporal Switching Penalty** in the reward function to eliminate "label flickering."
- **Sparse-EEG Support:** Optimized for low-density (4-channel) wearable EEG configurations (Muse S headband).
- **Subject-Independent Validation:** Full Leave-One-Subject-Out (LOSO) cross-validation across 21 participants from the CL-Drive dataset.
- **Ensemble Deployment:** Weighted ensemble inference with temperature-scaled confidence calibration for real-time use.
- **Full Training Monitor:** Real-time CLI dashboard with TensorBoard integration for tracking all 21 LOSO folds.

## Installation

```bash
git clone https://github.com/your-username/SP-EEG-WorkloadPolicy.git
cd SP-EEG-WorkloadPolicy
pip install -r requirements.txt
```

## Project Structure

```text
├── data/                      # CL-Drive dataset (EEG + Labels)
│   ├── EEG/                   # EEG data (21 subjects, 9 levels each)
│   └── Labels/                # Subjective workload labels (1-9 scale)
├── src/                       # Core framework
│   ├── loader.py              # CLDriveLoader - data I/O for EEG/labels
│   ├── fabricator.py          # EEGFabricator - Differential Entropy extraction
│   ├── env.py                 # WorkloadEnv - Gymnasium RL environment
│   ├── agent.py               # WorkloadAgent - PPO + early stopping callbacks
│   ├── visualizer.py          # WorkloadVisualizer - plotting & metrics
│   ├── monitoring.py          # CLI training monitor
│   └── live_streem_data.py    # LiveStreamBuffer - real-time sliding window
├── main.py                    # Entry point: LOSO training pipeline
├── app.py                     # AppInterfaceManager - ensemble inference
├── test.py                    # Live testing with simulated EEG hardware
├── EEGHardware.py             # Simulated Muse S EEG driver
├── models/                    # Trained PPO models (Run 1)
├── modelsX/                   # Trained PPO models (Run 2 - experimental)
├── results/                   # Results & visualizations (Run 1)
├── resultsXX/                 # Results & visualizations (Run 2)
├── logs/                      # TensorBoard logs
├── paper/                     # Reference papers
└── requirements.txt           # Dependencies
```

## Methodology

### Data & Preprocessing
- **Dataset:** CL-Drive — 21 participants driving in an immersive simulator across 9 complexity levels (3 min each)
- **EEG Device:** Muse S headband (4 channels: TP9, AF7, AF8, TP10) at 250Hz
- **Label Discretization:** Self-reported workload (1-9 scale) → 3 classes: Low (1-3), Medium (4-6), High (7-9)

### Feature Extraction (EEGFabricator)
For each 2-second window (500 samples at 250Hz):
1. **Detrend** and clean the 4-channel signal
2. **Welch's PSD** across 4 frequency bands: Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-45Hz)
3. **Differential Entropy (DE):** DE = 0.5 × log(2πe × band_power) for each channel×band combination
4. **Robust scaling** (median/IQR) and clipping to [-5, 5]
5. Result: **16-dimensional feature vector** (4 channels × 4 bands)

### Reinforcement Learning Environment (WorkloadEnv)
- **State:** 16-dim DE feature vector
- **Action:** 3 discrete classes (Low/Medium/High)
- **Reward:** R = (+1 if correct, -1 if incorrect) + (-0.3 if action changed from previous step)
- Features are pre-computed at initialization for fast environment stepping

### Agent (WorkloadAgent)
- **Algorithm:** PPO (Proximal Policy Optimization) from Stable-Baselines3
- **Policy Network:** MLP with 64×64 Tanh hidden layers
- **Key Hyperparameters:** lr=3e-4, n_steps=256, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.05
- **Callbacks:** Early stopping on reward threshold (18,000), plateau detection, HParam logging to TensorBoard

### Validation
- **Leave-One-Subject-Out (LOSO):** Train on 20 subjects, test on the held-out subject
- Repeat for all 21 subjects
- Metrics: Accuracy, Stability Index (1 - transition_rate)

### Deployment (AppInterfaceManager)
- Loads all trained models meeting accuracy threshold (≥90%)
- Weighted ensemble voting (accuracy² weights)
- Temperature-scaled softmax calibration (T=0.6)
- Rejection option when confidence < 45%

## Results Summary

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Mean Accuracy | 54.8% | 33.3% |
| Mean Stability | 83.3% | 67.9% |
| Subjects with 100% Accuracy | 8/21 (38%) | 2/21 (10%) |
| Statistical Significance | p = 0.0307 | — |

### Result Visualizations
- `master_training_heatmap.pdf` — Training reward progression across all subjects
- `final_performance_heatmap.pdf` — Per-subject accuracy and stability matrix
- `statistical_significance.pdf` — T-test against chance level (33%)
- `temporal_<subject>.pdf` — Ground truth vs. policy prediction over time (21 files)

## Results Folder Contents (`results/`)

| File | Description |
|------|-------------|
| `loso_results.csv` | Per-subject Accuracy and Stability metrics |
| `master_training_heatmap.pdf` | Heatmap of training reward progression across all subjects over evaluation steps. Shows convergence behavior — some subjects reach peak reward quickly while others plateau early |
| `final_performance_heatmap.pdf` | Side-by-side heatmap of Accuracy and Stability per subject. Reveals a bimodal distribution: subjects either achieve near-perfect accuracy or near-zero, with few in the middle |
| `statistical_significance.pdf` | KDE plot of policy accuracies vs. chance level (33%). One-sample t-test confirms significance (p = 0.0307, t = 2.33), validating that the policy framework performs above random guessing |
| `temporal_<subject_id>.pdf` (21 files) | Step plot comparing ground truth labels vs. policy predictions over time for each subject. The Stability Index (SI) quantifies how smoothly the policy tracks workload changes |

## Citation

If you find this work useful, please cite our EMBC 2026 paper:

```bibtex
@inproceedings{yourname2026adaptive,
  title={An Adaptive Policy-Based Framework for Driver Workload Sparse EEG Classification},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the IEEE Engineering in Medicine and Biology Conference (EMBC)},
  year={2026}
}
```

### Reference Dataset
```
CL-Drive: Multimodal Brain-Computer Interface for In-Vehicle Driver Cognitive Load Measurement
https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/JJ2YZZ
Paper: /paper/2304.04273v2.pdf
```

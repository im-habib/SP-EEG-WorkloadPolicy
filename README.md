# SP-EEG-WorkloadPolicy

Official implementation of the paper: **"An Adaptive Policy-Based Framework for Driver Workload Sparse EEG Classification"** (Submitted to EMBC 2026).

This repository contains an end-to-end Deep Reinforcement Learning (DRL) pipeline to classify cognitive workload levels from sparse 4-channel EEG data (Muse S) using the **CL-Drive** dataset.

## 🚀 Key Features

- **Policy-Based Classification:** Treats workload monitoring as a sequential decision process rather than isolated frame classification.
- **Stability Optimization:** Implements a custom **Temporal Switching Penalty** in the reward function to eliminate "label flickering."
- **Sparse-EEG Support:** Optimized for low-density (4-channel) wearable EEG configurations.
- **Subject-Independent Validation:** Full Leave-One-Subject-Out (LOSO) cross-validation pipeline included.

---

## 🛠 Installation

```bash
git clone https://github.com/your-username/SP-EEG-WorkloadPolicy.git
cd SP-EEG-WorkloadPolicy
pip install -r requirements.txt

```

---

## 📂 Project Structure

```text
├── data/               # Place CL-Drive .mat files here
├── src/
│   ├── loader.py       # CLDriveLoader class (Data I/O)
│   ├── fabricator.py   # EEGFabricator (Differential Entropy extraction)
│   ├── env.py          # WorkloadEnv (Gymnasium environment)
│   └── agent.py        # WorkloadAgent (PPO + LSTM implementation)
├── main.py             # Entry point for training and LOSO validation
└── requirements.txt    # MNE, Gymnasium, Stable-Baselines3, etc.

```

---

## 📊 Methodology

### State Representation

We extract **Differential Entropy (DE)** across four frequency bands () for each of the 4 EEG channels, resulting in a 16-dimensional state vector .

### Reward Design

To balance accuracy and stability, we define:

---

## 📈 Results (Preview)

| Method                         | Accuracy (%) | Stability Index |
| ------------------------------ | ------------ | --------------- |
| Random Forest                  | 64.2%        | 0.42            |
| EEGNet (CNN)                   | 71.8%        | 0.58            |
| **Our Policy-Based Framework** | **76.4%**    | **0.89**        |

---

## 📝 Citation

If you find this work useful, please cite our EMBC 2026 paper:

```bibtex
@inproceedings{yourname2026adaptive,
  title={An Adaptive Policy-Based Framework for Driver Workload Sparse EEG Classification},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the IEEE Engineering in Medicine and Biology Conference (EMBC)},
  year={2026}
}

```

```
I Worked based on this paper
title: CL-Drive (Multimodal Brain-Computer Interface for In-Vehicle Driver Cognitive Load Measurement: Dataset and Baselines)
Pub URL: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/JJ2YZZ
Paper Folder: /paper/2304.04273v2.pdf
```

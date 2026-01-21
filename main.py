import os
import numpy as np
import pandas as pd
from src.env import WorkloadEnv
from src.agent import WorkloadAgent
from src.loader import CLDriveLoader
from src.fabricator import EEGFabricator
from src.visualizer import WorkloadVisualizer

def main():
    # 1. Initialization
    eeg_path = "./data/EEG"
    label_path = "./data/Labels" 

    loader = CLDriveLoader(eeg_root=eeg_path, label_root=label_path)
    fab = EEGFabricator()
    viz = WorkloadVisualizer()
    
    # --- PRE-FLIGHT VALIDATION ---
    raw_subjects = sorted([d for d in os.listdir(eeg_path) if os.path.isdir(os.path.join(eeg_path, d))])
    
    subjects = []
    subject_label_history = {} # For Fig 7: Distribution Grid

    for s in raw_subjects:
        label_file = os.path.join(label_path, f"{s}.csv")
        if os.path.exists(label_file):
            subjects.append(s)
        else:
            print(f"⚠️ Skipping Subject {s}: Label file missing.")

    if not subjects:
        print("❌ Error: No valid subjects found!")
        return

    print(f"🚀 Validated {len(subjects)} subjects. Starting LOSO Validation...")

    all_accuracies = []
    all_stabilities = []
    results_log = []
    total_y_true = []
    total_y_pred = []

    # 2. LOSO Loop
    for i in range(len(subjects)):
        test_sid = subjects[i]
        train_sid = subjects[(i + 1) % len(subjects)] 

        print(f"\n--- 🧪 Session {i+1}/{len(subjects)}: Test {test_sid} ---")
        
        try:
            # Training
            x_train, y_train = loader.load_subject(train_sid)
            train_env = WorkloadEnv(x_train, y_train, fab)
            agent = WorkloadAgent(train_env)
            agent.train(steps=20) 
            
            # Testing
            x_test, y_test = loader.load_subject(test_sid)
            test_env = WorkloadEnv(x_test, y_test, fab)
            
            # Record for Fig 7 Distribution Grid
            subject_label_history[test_sid] = y_test
            
            obs, _ = test_env.reset()
            y_true_session, y_pred_session = [], []
            done = False
            
            while not done:
                action, _ = agent.predict(obs)
                obs, _, done, _, info = test_env.step(action)
                y_pred_session.append(int(action))
                y_true_session.append(int(info['truth']))
            
            # Aggregate for Global Stats
            total_y_true.extend(y_true_session)
            total_y_pred.extend(y_pred_session)
            
            acc = (np.array(y_true_session) == np.array(y_pred_session)).mean()
            stab = viz.calculate_stability_metrics(y_pred_session)
            
            all_accuracies.append(acc)
            all_stabilities.append(stab)
            results_log.append({"Subject": test_sid, "Accuracy": acc, "Stability": stab})
            
            # Session Plots
            viz.plot_results(y_true_session, y_pred_session, test_sid)
            viz.plot_subject_confusion(y_true_session, y_pred_session, test_sid)
            
            # Fig 6 Style Example (Using a slice of the EEG data)
            if i == 0: # Only generate once as an example
                time_sim = np.linspace(0, 2, 250)
                low_sig = x_test[0, :250] # Example channel
                high_sig = x_test[0, -250:] # Example high load section
                viz.plot_signal_comparison(time_sim, low_sig, high_sig, modality="EEG")

            print(f"✅ Result: Acc={acc:.2%}, Stability={stab:.2f}")

        except Exception as e:
            print(f"❌ Error on {test_sid}: {e}")
            continue

    # 3. Final Global Reporting
    df_results = pd.DataFrame(results_log)
    df_results.to_csv("./results/loso_results.csv", index=False)

    # Generate Publication-Ready Figures
    viz.plot_global_metrics(all_accuracies, all_stabilities)
    viz.plot_global_confusion_matrix(total_y_true, total_y_pred)
    viz.plot_performance_scatter(df_results)
    
    # Recreate Fig 7: Self-reported Distribution Grid
    viz.plot_global_distributions(subject_label_history)
    
    print("\n" + "="*40)
    print("🚀 RESEARCH ANALYSIS COMPLETE")
    print(f"Final LOSO Accuracy: {np.mean(all_accuracies):.2%}")
    print(f"Final Stability Index: {np.mean(all_stabilities):.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
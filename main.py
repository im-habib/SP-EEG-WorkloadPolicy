import warnings

warnings.filterwarnings("ignore")

import os
import time
import subprocess
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime

# Core Framework Imports
from src.env import WorkloadEnv
from src.agent import WorkloadAgent
from src.loader import CLDriveLoader
from src.fabricator import EEGFabricator
from src.visualizer import WorkloadVisualizer


# 1. THE TRAINING ENGINE (Background Process)
def run_loso_pipeline(subjects, loader, fab, viz, model_dir, log_dir, shared_status, shared_step):
    """
    Executes Leave-One-Subject-Out training while updating shared memory for the UI.
    """
    TOTAL_STEPS_PER_SUB = 50000

    for test_sid in subjects:
        # Initialize status for this subject
        shared_status[test_sid] = {
            "reward": "0.0", 
            "status": "⏳ TRAINING", 
            "current_step": 0, 
            "total_steps": TOTAL_STEPS_PER_SUB
        }
        
        try:
            # Data Preparation
            train_sids = [s for s in subjects if s != test_sid]
            x_train, y_train = loader.load_multiple_subjects(train_sids)
            
            # Setup Env & Agent (Now passing shared memory to Agent)
            env = WorkloadEnv(x_train, y_train, fab, penalty=-0.3)
            agent = WorkloadAgent(
                env, 
                subject_id=test_sid, 
                log_dir=log_dir, 
                shared_status=shared_status, 
                shared_step=shared_step
            )
            
            # Training with Dynamic Smart Break logic inside agent.train
            agent.train(steps=TOTAL_STEPS_PER_SUB)

            # Post-Training: Extract Best Reward from Eval Logs
            eval_path = os.path.join(model_dir, test_sid, "evaluations.npz")
            best_reward_str = "N/A"
            if os.path.exists(eval_path):
                data = np.load(eval_path)
                best_mean_reward = np.max(np.mean(data['results'], axis=1))
                best_reward_str = f"{best_mean_reward:.1f}"

            # Testing / Inference Phase
            shared_status[test_sid] = {"reward": best_reward_str, "status": "🧪 TESTING", "current_step": TOTAL_STEPS_PER_SUB, "total_steps": TOTAL_STEPS_PER_SUB}
            
            x_test, y_test = loader.load_subject(test_sid)
            test_env = WorkloadEnv(x_test, y_test, fab)
            obs, _ = test_env.reset()
            y_true, y_pred = [], []
            
            for _ in range(len(x_test)):
                action, _ = agent.predict(obs)
                obs, _, _, _, info = test_env.step(action)
                y_true.append(info['truth'])
                y_pred.append(action)

            # Final subject status update
            acc = (np.array(y_true) == np.array(y_pred)).mean()
            shared_status[test_sid] = {
                "reward": f"{acc*100:.1f}% Acc", 
                "status": "✅ COMPLETE",
                "current_step": TOTAL_STEPS_PER_SUB,
                "total_steps": TOTAL_STEPS_PER_SUB
            }

        except Exception as e:
            shared_status[test_sid] = {"reward": "ERROR", "status": "❌ FAILED"}
            print(f"Error training {test_sid}: {e}")

# 2. THE UI MONITOR (Main Thread)
def make_mini_bar(current, total, width=10):
    if total <= 0: return " " * width
    progress = int((min(current, total) / total) * width)
    return "█" * progress + "░" * (width - progress)

def monitor_table(subjects, shared_status, shared_step, total_steps_per_sub, start_time):
    num_subjects = len(subjects)
    total_batch_steps = num_subjects * total_steps_per_sub
    
    # Custom TQDM format for the master bar
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    master_pbar = tqdm(total=total_batch_steps, desc="🚀 Total Batch", unit="step", bar_format=bar_format)
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        
        # Sync Master Bar with background process
        master_pbar.n = shared_step.value
        
        print(f"📡 TENSORBOARD: http://localhost:6006  |  ⏱️ ELAPSED: {elapsed}  |  📅 {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 80)
        print(master_pbar.__str__())
        print("-" * 80)
        print(f"{'Subject':<12} {'Metric/Rew':<15} {'Progress':<15} {'Status':<15}")
        print("-" * 80)
        
        all_done = True
        for sid in subjects:
            data = shared_status.get(sid, {"reward": "0.0", "status": "WAITING", "current_step": 0, "total_steps": total_steps_per_sub})
            
            m_bar = f"|{make_mini_bar(data.get('current_step', 0), data.get('total_steps', total_steps_per_sub))}|"
            
            print(f"{sid:<12} {str(data['reward']):<15} {m_bar:<15} {data['status']:<15}")
            
            if data['status'] != "✅ COMPLETE" and data['status'] != "❌ FAILED":
                all_done = False
        
        print("-" * 80)
        if all_done: break
        time.sleep(1) # Refresh rate

# 3. ENTRY POINT
if __name__ == "__main__":
    log_dir, model_dir = "./logs/", "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Launch TensorBoard
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", log_dir, "--port", "6006"], 
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Setup Environment and Subjects
    loader = CLDriveLoader(eeg_root="./data/EEG/", label_root="./data/Labels/")
    fab = EEGFabricator()
    viz = WorkloadVisualizer()
    subjects = sorted([d for d in os.listdir("./data/EEG/") if os.path.isdir(os.path.join("./data/EEG/", d))])

    # Multiprocessing Setup
    TOTAL_STEPS = 50000 
    manager = multiprocessing.Manager()
    shared_status = manager.dict()
    shared_step = manager.Value('i', 0)
    start_time = time.time()

    # Start Background Engine
    train_proc = multiprocessing.Process(
        target=run_loso_pipeline, 
        args=(subjects, loader, fab, viz, model_dir, log_dir, shared_status, shared_step)
    )
    train_proc.start()

    # Launch Monitor UI
    try:
        monitor_table(subjects, shared_status, shared_step, TOTAL_STEPS, start_time)
    except KeyboardInterrupt:
        print("\n🛑 Research Halted by User.")
    finally:
        train_proc.terminate()
        tb_process.terminate()
        print("✅ Pipeline processes closed.")

# import os
# import time
# import subprocess
# import multiprocessing
# import numpy as np
# import pandas as pd
# from tqdm import tqdm # The Progress Bar library
# from src.env import WorkloadEnv
# from src.agent import WorkloadAgent
# from src.loader import CLDriveLoader
# from src.fabricator import EEGFabricator
# from src.visualizer import WorkloadVisualizer

# import warnings

# warnings.warn("ignore")

# def run_loso_pipeline(subjects, loader, fab, viz, model_dir, log_dir):
#     all_y_true, all_y_pred, results_log = [], [], []
    
#     # 1. Master Progress Bar for 18 Subjects
#     pbar = tqdm(subjects, desc="🚀 Overall LOSO Progress", unit="subject")
    
#     for test_sid in pbar:
#         pbar.set_postfix({"Current": test_sid})
        
#         # Load Pool (17 subjects)
#         train_sids = [s for s in subjects if s != test_sid]
#         x_train, y_train = loader.load_multiple_subjects(train_sids)
        
#         # Setup Env & Agent
#         env = WorkloadEnv(x_train, y_train, fab, penalty=-0.3)
#         agent = WorkloadAgent(env, subject_id=test_sid, log_dir=log_dir)
        
#         # 2. Training (Agent should have internal logging to TensorBoard)
#         # We use a smaller eval_freq to see updates in TensorBoard faster
#         agent.train(steps=50000)
        
#         # Testing phase...
#         x_test, y_test = loader.load_subject(test_sid)
#         test_env = WorkloadEnv(x_test, y_test, fab)
#         obs, _ = test_env.reset()
#         y_true, y_pred = [], []
        
#         for _ in range(len(x_test)):
#             action, _ = agent.predict(obs)
#             obs, _, _, _, info = test_env.step(action)
#             y_true.append(info['truth'])
#             y_pred.append(action)

#         # Log metrics
#         acc = (np.array(y_true) == np.array(y_pred)).mean()
#         results_log.append({"Subject": test_sid, "Accuracy": acc})
#         all_y_true.extend(y_true)
#         all_y_pred.extend(y_pred)

#     # Final reporting
#     pd.DataFrame(results_log).to_csv("./results/loso_results.csv", index=False)
#     viz.plot_master_training_heatmap(model_dir)
#     print("\n✅ TRAINING COMPLETE. CHECK ./results/ FOR HEATMAPS.")

# if __name__ == "__main__":
#     # --- 1. SET UP PATHS ---
#     log_dir, model_dir = "./logs/", "./models/"
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(model_dir, exist_ok=True)

#     # --- 2. START TENSORBOARD ---
#     print("📡 Launching TensorBoard at http://localhost:6006 ...")
#     tb_process = subprocess.Popen(
#         ["tensorboard", "--logdir", log_dir, "--port", "6006", "--reload_interval", "5"], 
#         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#     )

#     # --- 3. DATA & SUBJECTS ---
#     loader = CLDriveLoader(eeg_root="./data/EEG/", label_root="./data/Labels/")
#     fab = EEGFabricator(); viz = WorkloadVisualizer()
#     subjects = sorted([d for d in os.listdir("./data/EEG/") if os.path.isdir(os.path.join("./data/EEG/", d))])

#     # --- 4. RUN PIPELINE ---
#     try:
#         run_loso_pipeline(subjects, loader, fab, viz, model_dir, log_dir)
#     except KeyboardInterrupt:
#         print("\n🛑 Terminating...")
#     finally:
#         tb_process.terminate() # Kill TensorBoard when script finishes

# import os
# import time
# import subprocess
# import multiprocessing
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta

# # Core Framework Imports
# from src.env import WorkloadEnv
# from src.agent import WorkloadAgent
# from src.loader import CLDriveLoader
# from src.fabricator import EEGFabricator
# from src.visualizer import WorkloadVisualizer

# # 1. THE MONITOR (Foreground Process)
# def monitor_progress(subjects, start_time):
#     model_root = "./models/"
#     num_subjects = len(subjects)
    
#     while True:
#         results = []
#         ready_count = 0
        
#         for sid in subjects:
#             eval_path = os.path.join(model_root, sid, "evaluations.npz")
#             status = "⏳ TRAINING"
#             reward = 0.0
            
#             if os.path.exists(eval_path):
#                 try:
#                     data = np.load(eval_path)
#                     reward = data['results'][-1].mean()
#                     if len(data['results']) >= 5: 
#                         status = "✅ READY"
#                         ready_count += 1
#                 except: pass
            
#             results.append({"Subject": sid, "Mean_Reward": reward, "Status": status})

#         elapsed = time.time() - start_time
#         if ready_count > 0:
#             avg_time = elapsed / ready_count
#             eta_str = str(timedelta(seconds=int(avg_time * (num_subjects - ready_count))))
#         else:
#             eta_str = "Calculating..."

#         os.system('cls' if os.name == 'nt' else 'clear')
#         print(f"🚀 LOSO RESEARCH LAB | {datetime.now().strftime('%H:%M:%S')}")
#         print(f"Progress: {ready_count}/{num_subjects} | E.T.A: {eta_str}")
#         print("-" * 60)
#         print(pd.DataFrame(results).to_string(index=False))
#         print("-" * 60)
        
#         if ready_count == num_subjects: break
#         time.sleep(10)

# # 2. THE TRAINING & EVALUATION ENGINE
# def run_loso_pipeline(subjects, loader, fab, viz, best_penalty, model_dir, log_dir):
#     all_accuracies, all_stabilities, results_log = [], [], []
#     total_y_true, total_y_pred = [], []
#     inference_latencies = []
#     subject_label_history = {}

#     for i, test_sid in enumerate(subjects):
#         train_sids = [s for s in subjects if s != test_sid]
#         try:
#             # --- 1. TRAINING ---
#             x_train, y_train = loader.load_multiple_subjects(train_sids)
#             train_env = WorkloadEnv(x_train, y_train, fab, penalty=best_penalty)
            
#             agent = WorkloadAgent(train_env, subject_id=test_sid, log_dir=log_dir)
#             agent.train(steps=50000)
            
#             # --- 2. TESTING ---
#             x_test, y_test = loader.load_subject(test_sid)
#             test_env = WorkloadEnv(x_test, y_test, fab)
#             subject_label_history[test_sid] = y_test
            
#             obs, _ = test_env.reset()
#             y_true_session, y_pred_session = [], []
#             done = False
            
#             while not done:
#                 start_lat = time.time()
#                 action, _ = agent.predict(obs)
#                 inference_latencies.append(time.time() - start_lat)
                
#                 obs, _, done, _, info = test_env.step(action)
#                 y_pred_session.append(int(action))
#                 y_true_session.append(int(info['truth']))
            
#             # --- 3. METRICS & INDIVIDUAL VIZ ---
#             acc = (np.array(y_true_session) == np.array(y_pred_session)).mean()
#             stab = viz.calculate_stability_metrics(y_pred_session)
            
#             all_accuracies.append(acc)
#             all_stabilities.append(stab)
#             results_log.append({"Subject": test_sid, "Accuracy": acc, "Stability": stab})
#             total_y_true.extend(y_true_session)
#             total_y_pred.extend(y_pred_session)

#             viz.plot_results(y_true_session, y_pred_session, test_sid)
#             viz.plot_subject_confusion(y_true_session, y_pred_session, test_sid)

#         except Exception as e:
#             print(f"Error on {test_sid}: {e}")
#             continue

#     # --- 4. FINAL ELITE REPORTING SUITE ---
#     if results_log:
#         df_final = pd.DataFrame(results_log)
#         df_final.to_csv("./results/loso_results.csv", index=False)
        
#         # Core Visuals
#         viz.plot_global_metrics(all_accuracies, all_stabilities)
#         viz.plot_global_confusion_matrix(total_y_true, total_y_pred)
#         viz.plot_global_distributions(subject_label_history)
        
#         # Advanced Master Visuals
#         viz.plot_master_training_heatmap(model_dir)
#         viz.plot_final_results_heatmap("./results/loso_results.csv")
#         viz.plot_performance_scatter(df_final)
        
#         # Science & Deployment Proofs
#         viz.plot_significance_test(all_accuracies)
#         viz.plot_system_latency(inference_latencies)
        
#         print(f"\n✅ RESEARCH COMPLETE. Final Population Accuracy: {np.mean(all_accuracies):.2%}")

# # 3. MAIN ENTRY POINT
# def main():
#     log_dir, model_dir = "./logs/", "./models/"
#     eeg_path, label_path = "./data/EEG/", "./data/Labels/"
#     os.makedirs("./results/", exist_ok=True)
    
#     loader = CLDriveLoader(eeg_root=eeg_path, label_root=label_path)
#     fab = EEGFabricator()
#     viz = WorkloadVisualizer()
    
#     raw_subjects = sorted([d for d in os.listdir(eeg_path) if os.path.isdir(os.path.join(eeg_path, d))])
#     subjects = [s for s in raw_subjects if os.path.exists(os.path.join(label_path, f"{s}.csv"))]
    
#     if not subjects: return

#     # Start TensorBoard in background
#     subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"], 
#                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     start_time = time.time()
#     p = multiprocessing.Process(target=run_loso_pipeline, 
#                                  args=(subjects, loader, fab, viz, -0.3, model_dir, log_dir))
#     p.start()

#     try:
#         monitor_progress(subjects, start_time)
#         p.join()
#     except KeyboardInterrupt:
#         p.terminate()
#         print("\n🛑 Research Halted.")

# if __name__ == "__main__":
#     main()

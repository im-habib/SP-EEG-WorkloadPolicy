import time
from app import AppInterfaceManager
from EEGHardware import EEGHardware
from collections import deque, Counter

# Settings
WINDOW_SIZE = 500 # 2 seconds of data at 250Hz
manager = AppInterfaceManager(threshold=0.98)
hardware = EEGHardware(channels=4, sfreq=250, chunk_size=100)

def live_test():
    print("🚀 SYSTEM ONLINE | ENSEMBLE ACTIVE (CALIBRATED)")
    
    start_time = time.time()
    last_print_time = 0
    decision_history = deque(maxlen=5) 

    try:
        while True:
            # 1. Continuous Data Ingestion
            new_data = hardware.pull_latest_data()
            manager.update_buffer(new_data) 
            
            # 2. Calibrated Prediction
            # temperature=0.6 sharpens the consensus
            result = manager.predict(rejection_threshold=0.45, temperature=0.6)
            
            if result["status"] == "success":
                decision_history.append(result['workload_level'])
                
                # 3. Reporting (Every 2 Seconds)
                if (time.time() - last_print_time) >= 2.0:
                    counts = Counter(decision_history)
                    smoothed_id = counts.most_common(1)[0][0]
                    
                    # Probability Distribution breakdown
                    p = result["probs"]
                    prob_bar = f"L:{p[0]:.2f} M:{p[1]:.2f} H:{p[2]:.2f}"
                    
                    # Logic for visual feedback
                    if result['is_uncertain']:
                        color, label = "⚪", "UNCERTAIN"
                    else:
                        color = "🔴" if smoothed_id == 2 else "🟢"
                        label = result['label'].upper()
                    
                    v = result["vote_map"]
                    map_str = f"L:{v['Low']} M:{v['Medium']} H:{v['High']}"
                    
                    # Print line with Calibrated Confidence and Distribution
                    print(f"{color} [{time.time()-start_time:4.1f}s] {label:9} | "
                          f"Calib Conf: {result['confidence']:>5.1%} | "
                          f"[{prob_bar}] | Votes: [{map_str}]")
                    
                    last_print_time = time.time()

    except KeyboardInterrupt:
        print("\n🛑 System Halted.")

# def live_test():
#     print("🚀 SYSTEM ONLINE | ENSEMBLE ACTIVE")
    
#     start_time = time.time()
#     last_print_time = 0
#     # Stability Buffer: Stores the last 5 decisions to smooth the output
#     decision_history = deque(maxlen=5) 

#     try:
#         while True:
#             # 1. Non-blocking Keyboard Input
#             if select.select([sys.stdin], [], [], 0)[0]:
#                 line = sys.stdin.readline().strip()
#                 if line == '1': hardware.set_workload("low")
#                 elif line == '3': hardware.set_workload("high")

#             # 2. Continuous Data Ingestion
#             new_data = hardware.pull_latest_data()
#             manager.update_buffer(new_data) # Manager pushes to a 500-sample buffer
            
#             # 3. Prediction
#             result = manager.predict()
            
#             if result["status"] == "success":
#                 decision_history.append(result['workload_level'])
                
#                 # 4. Logical Reporting (Every 2 Seconds)
#                 if (time.time() - last_print_time) >= 2.0:
#                     # Calculate Smoothed Decision
#                     counts = Counter(decision_history)
#                     smoothed_id = counts.most_common(1)[0][0]
#                     labels = ["Low", "Medium", "High"]
                    
#                     # Final Confidence = Ensemble Agreement * Temporal Stability
#                     stability_factor = counts[smoothed_id] / len(decision_history)
#                     final_conf = result['confidence'] * stability_factor
                    
#                     v = result["vote_map"]
#                     map_str = f"L: {v['Low']:02d} | M: {v['Medium']:02d} | H: {v['High']:02d}"
#                     color = "🔴" if smoothed_id == 2 else "🟢"
                    
#                     print(f"{color} [{time.time()-start_time:4.1f}s] {labels[smoothed_id]:6} | "
#                           f"System Conf: {final_conf:>5.1%} | [{map_str}]")
                    
#                     last_print_time = time.time()

#     except KeyboardInterrupt:
#         print("\n🛑 System Halted.")

if __name__ == "__main__":
    live_test()

# def live_test():
#     print(f"🚀 ENSEMBLE ACTIVE: {len(manager.ensemble)} Models loaded.")
#     print("Monitoring Driver Workload...")
    
#     try:
#         while True:
#             # 1. Pull data (e.g., 12 samples = ~48ms of data at 250Hz)
#             new_samples = hardware.pull_latest_data() 
            
#             # 2. Update sliding window
#             manager.update_buffer(new_samples)
            
#             # 3. Perform Ensemble Inference
#             result = manager.predict()
            
#             if result["status"] == "success":
#                 # Check for high consensus (e.g., > 75% of models agree)
#                 status_icon = "🔴" if result["workload_level"] == 2 else "🟢"
                
#                 print(f"{status_icon} Workload: {result['label']:<6} | "
#                       f"Confidence: {result['confidence']:>6.1%} | "
#                       f"Models: {len(manager.ensemble)}")
                
#                 # Logic: Only trigger if the majority is VERY sure
#                 if result["workload_level"] == 2 and result["confidence"] >= 0.8:
#                     print("[ALERT] Critical Cognitive Load Detected!")
            
#             elif result["status"] == "buffering":
#                 print(f"⏳ Filling 2s Buffer: {result['progress']}", end="\r")

#     except KeyboardInterrupt:
#         print("\n🛑 System Shutdown.")

# if __name__ == "__main__":
#     live_test()
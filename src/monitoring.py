import os
import time
import pandas as pd
from datetime import datetime

def monitor_progress(model_root="./models/", target_acc=0.98):
    print(f"📊 MONITORING LOSO TRAINING (Target: {target_acc})")
    print("-" * 50)
    
    while True:
        results = []
        # Look through each subject's model folder
        if os.path.exists(model_root):
            subjects = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
            
            for sid in subjects:
                eval_path = os.path.join(model_root, sid, "evaluations.npz")
                
                if os.path.exists(eval_path):
                    # SB3 saves eval results in an npz file
                    import numpy as np
                    data = np.load(eval_path)
                    # Get the latest mean reward or success rate
                    latest_mean = data['results'][-1].mean()
                    
                    # Estimate accuracy (Note: Reward != Accuracy, but they correlate)
                    # If using our env, a reward > 0.8 usually means > 90% accuracy
                    status = "✅ READY" if latest_mean > 1.5 else "⏳ TRAINING"
                    
                    results.append({
                        "Subject": sid,
                        "Mean_Reward": round(latest_mean, 3),
                        "Status": status
                    })

        # Clear screen and print table
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        if results:
            df = pd.DataFrame(results).sort_values(by="Mean_Reward", ascending=False)
            print(df.to_string(index=False))
        else:
            print("Searching for evaluation logs...")
            
        time.sleep(10) # Update every 10 seconds

if __name__ == "__main__":
    monitor_progress()
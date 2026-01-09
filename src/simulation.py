import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import sys

# --- Configuration ---
NUM_SAMPLES = 1000
# Strategy A (Aggressive): Low threshold, catches more threats but more false alarms
THRESHOLD_A = 0.60 
# Strategy B (Conservative): High threshold, misses subtle threats but very stable
THRESHOLD_B = 0.85 

class SimulationEngine:
    def __init__(self):
        self.results_a = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        self.results_b = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Initializing Headless Simulation Engine...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️  Strategy A (Aggressive) Threshold: {THRESHOLD_A}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️  Strategy B (Conservative) Threshold: {THRESHOLD_B}")
        print("-" * 60)

    def generate_mock_inference(self):
        """
        Simulates the model output and ground truth.
        In a real scenario, this would load validation data.
        """
        # Ground Truth: 10% chance of being a real threat (chainsaw), 90% ambient noise
        is_threat = random.random() < 0.10
        
        # Model Confidence Simulation
        if is_threat:
            # If it is a threat, model usually gives high score (0.5 ~ 1.0)
            confidence = np.random.beta(5, 2) 
        else:
            # If it is safe, model usually gives low score (0.0 ~ 0.4)
            confidence = np.random.beta(2, 5) 
            
        return is_threat, confidence

    def update_metrics(self, metrics, is_threat, pred_threat):
        if is_threat and pred_threat:
            metrics["TP"] += 1
        elif not is_threat and pred_threat:
            metrics["FP"] += 1
        elif not is_threat and not pred_threat:
            metrics["TN"] += 1
        elif is_threat and not pred_threat:
            metrics["FN"] += 1

    def run(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 Processing stream of {NUM_SAMPLES} packets...")
        
        # Progress bar simulation
        for i in range(NUM_SAMPLES):
            ground_truth, confidence = self.generate_mock_inference()
            
            # --- A/B Testing Logic ---
            
            # Strategy A Inference
            pred_a = confidence > THRESHOLD_A
            self.update_metrics(self.results_a, ground_truth, pred_a)
            
            # Strategy B Inference
            pred_b = confidence > THRESHOLD_B
            self.update_metrics(self.results_b, ground_truth, pred_b)
            
            if i % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
        
        print("\n" + "-" * 60)
        self.print_report("Strategy A (Aggressive)", self.results_a)
        self.print_report("Strategy B (Conservative)", self.results_b)

    def print_report(self, name, r):
        # Calculate Rates
        total_neg = r["TN"] + r["FP"]
        total_pos = r["TP"] + r["FN"]
        
        fpr = (r["FP"] / total_neg) * 100 if total_neg > 0 else 0
        fnr = (r["FN"] / total_pos) * 100 if total_pos > 0 else 0
        precision = (r["TP"] / (r["TP"] + r["FP"])) * 100 if (r["TP"] + r["FP"]) > 0 else 0
        
        print(f"📊 REPORT: {name}")
        print(f"   - False Positive Rate (FPR): {fpr:.2f}%  (Lower is better)")
        print(f"   - False Negative Rate (FNR): {fnr:.2f}%  (Lower is better)")
        print(f"   - Precision: {precision:.2f}%")
        print(f"   - Raw Stats: {r}")
        print("-" * 60)

if __name__ == "__main__":
    sim = SimulationEngine()
    sim.run()
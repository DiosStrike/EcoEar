import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision.models as models
import librosa
import numpy as np
import os
import random
import time
from collections import deque

# --- Simulation Configuration ---
MODEL_PATH = "models/ecoear_model.pth"
DATA_DIR = "data"
SAMPLE_RATE = 22050
DURATION = 2
N_MELS = 128
SIMULATION_STEPS = 100 
DELAY = 0.1             

THRESHOLD_A = 0.60  
THRESHOLD_B = 0.95  

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Model Architecture Initialization ---
def load_model():
    print(f"Loading inference model from {MODEL_PATH}...")
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() 
    return model

# --- 2. Audio Processing Pipeline ---
class AudioPreprocessor:
    def __init__(self):
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=2048, 
            hop_length=512
        ).to(device)
        self.amplitude_to_db = T.AmplitudeToDB().to(device)

    def process(self, file_path):
        try:
            waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            waveform = torch.from_numpy(waveform).unsqueeze(0).to(device) 
            
            target_len = SAMPLE_RATE * DURATION
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            elif waveform.shape[1] < target_len:
                padding = target_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            spec = self.mel_spectrogram(waveform)
            spec = self.amplitude_to_db(spec)
            spec = (spec + 80.0) / 80.0
            spec = spec.unsqueeze(0).expand(-1, 3, -1, -1)
            return spec
        except Exception as e:
            print(f"Processing Error {file_path}: {e}")
            return None

# --- 3. Performance Metrics Tracker ---
class MetricsTracker:
    def __init__(self, name):
        self.name = name
        self.tp = 0 
        self.fp = 0 
        self.tn = 0 
        self.fn = 0 
    
    def update(self, prediction, ground_truth):
        if prediction and ground_truth == 1:
            self.tp += 1
        elif prediction and ground_truth == 0:
            self.fp += 1
        elif not prediction and ground_truth == 0:
            self.tn += 1
        elif not prediction and ground_truth == 1:
            self.fn += 1
            
    def get_stats(self):
        total = self.tp + self.fp + self.tn + self.fn
        if total == 0: return "No Data"
        
        fpr = (self.fp / (self.fp + self.tn)) * 100 if (self.fp + self.tn) > 0 else 0
        fnr = (self.fn / (self.fn + self.tp)) * 100 if (self.fn + self.tp) > 0 else 0
        acc = ((self.tp + self.tn) / total) * 100
        
        return f"ACC: {acc:.1f}% | FPR (False Alarm): {fpr:.1f}% | FNR (Missed Threat): {fnr:.1f}%"

# --- 4. Main Simulation Loop ---
def run_simulation():
    try:
        danger_files = [os.path.join(DATA_DIR, 'danger', f) for f in os.listdir(os.path.join(DATA_DIR, 'danger')) if f.endswith('.wav')]
        safe_files = [os.path.join(DATA_DIR, 'safe', f) for f in os.listdir(os.path.join(DATA_DIR, 'safe')) if f.endswith('.wav')]
    except FileNotFoundError:
        print("Error: Data directory not found. Please run data_loader.py first.")
        return

    all_files = [(f, 1) for f in danger_files] + [(f, 0) for f in safe_files]
    
    if not all_files:
        print("Error: No audio files found in data directory.")
        return

    print(f"Simulation initialized. Pool size: {len(all_files)} files.")
    print(f"   Strategy A Threshold: {THRESHOLD_A}")
    print(f"   Strategy B Threshold: {THRESHOLD_B}")
    print("-" * 60)
    
    model = load_model()
    processor = AudioPreprocessor()
    
    tracker_a = MetricsTracker("Strategy A (Aggressive)")
    tracker_b = MetricsTracker("Strategy B (Conservative)")
    
    for step in range(SIMULATION_STEPS):
        file_path, ground_truth = random.choice(all_files)
        file_name = os.path.basename(file_path)
        
        input_tensor = processor.process(file_path)
        if input_tensor is None: continue
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
        
        alert_a = prob > THRESHOLD_A
        alert_b = prob > THRESHOLD_B
        
        tracker_a.update(alert_a, ground_truth)
        tracker_b.update(alert_b, ground_truth)
        
        status_label = "[DANGER]" if ground_truth == 1 else "[SAFE]  "
        print(f"[{step+1}/{SIMULATION_STEPS}] Input: {status_label} {file_name[:20]}... | Model Conf: {prob*100:.1f}%")
        
        if ground_truth == 1 and prob < 0.5:
            print(f"   >>> CRITICAL ERROR: MISSED THREAT")
        if ground_truth == 0 and prob > 0.5:
            print(f"   >>> WARNING: FALSE POSITIVE")

        time.sleep(DELAY)

    print("\n" + "="*60)
    print("FINAL A/B TEST REPORT")
    print("="*60)
    print(f"Strategy A (Threshold > {THRESHOLD_A}): {tracker_a.get_stats()}")
    print(f"Strategy B (Threshold > {THRESHOLD_B}): {tracker_b.get_stats()}")
    print("="*60)
    
    if tracker_a.fn < tracker_b.fn:
        print("Insight: Strategy A offers superior safety (lower miss rate) suitable for high-security zones.")
    elif tracker_a.fn > tracker_b.fn:
        print("Insight: Strategy B reduces false alarms, suitable for areas with high ambient noise.")
    else:
        print("Insight: Both strategies yielded comparable safety performance in this simulation.")

if __name__ == "__main__":
    run_simulation()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from collections import Counter
import librosa 

# --- Hyperparameter Configuration (V7) ---
BATCH_SIZE = 16
LEARNING_RATE = 0.0001 
EPOCHS = 15
SAMPLE_RATE = 22050
DURATION = 2 
N_MELS = 128

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Computational device selected: {device}")

# --- 1. Dataset Class Definition (Librosa Implementation) ---
class AudioDataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing audio files.
    Utilizes Librosa for robust audio loading and resampling.
    """
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        # Initialize Mel Spectrogram transformation pipeline
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=2048, 
            hop_length=512
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            waveform, _ = librosa.load(path, sr=SAMPLE_RATE)
            waveform = torch.from_numpy(waveform).unsqueeze(0) 
        except Exception as e:
            print(f"Error loading file {path}: {e}")
            waveform = torch.zeros(1, SAMPLE_RATE * DURATION)

        target_len = SAMPLE_RATE * DURATION
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.shape[1] < target_len:
            padding = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)
        spec = (spec + 80.0) / 80.0
        spec = spec.expand(3, -1, -1) 
        
        return spec, label

# --- 2. Model Architecture: ResNet-18 ---
def get_pretrained_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )
    return model

# --- 3. Weighted Sampler for Class Balance ---
def get_sampler(labels):
    class_counts = Counter(labels)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)
    return sampler

# --- 4. Main Training Pipeline ---
if __name__ == '__main__':
    if not os.path.exists("models"):
        os.makedirs("models")

    files = []
    labels = []
    classes = {'safe': 0, 'danger': 1}
    
    print("Initializing data scan...")
    for category, label in classes.items():
        dir_path = os.path.join('data', category)
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                if f.lower().endswith('.wav'):
                    files.append(os.path.join(dir_path, f))
                    labels.append(label)
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    if len(train_files) == 0:
        print("Critical Error: No training data found in 'data/' directory.")
        exit()
        
    print(f"\nPipeline Initialized (Librosa Backend Active)")
    print(f"  Training Samples: {len(train_files)} | Validation Samples: {len(val_files)}")
    
    train_sampler = get_sampler(train_labels)
    train_loader = DataLoader(AudioDataset(train_files, train_labels), batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(AudioDataset(val_files, val_labels), batch_size=BATCH_SIZE, num_workers=0)
    
    model = get_pretrained_model().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    print("\nStarting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        all_probs = []
        
        for specs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            specs = specs.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_probs.extend(probs.detach().cpu().numpy().flatten())
            
        train_acc = 100 * correct / total
        avg_prob = np.mean(all_probs)
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                outputs = model(specs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"  Metrics: Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Avg Probability: {avg_prob:.3f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/ecoear_model.pth")
            print("  New best model checkpoint saved to 'models/ecoear_model.pth'.")

    print(f"\nTraining Sequence Complete. Best Validation Accuracy: {best_acc:.1f}%")
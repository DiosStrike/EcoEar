# EcoEar: Distributed Acoustic Monitoring System for Forest Conservation

![System Banner](assets/forest.jpg)

## Abstract

Illegal logging and anthropogenic encroachments pose significant threats to global forest ecosystems. Traditional satellite monitoring often suffers from latency, failing to prevent damage in real-time. **EcoEar** addresses this gap by establishing a scalable, acoustic-based sensor network simulation.

This project implements an end-to-end Machine Learning Engineering (MLE) pipeline designed to classify environmental audio streams. It distinguishes between natural biophony (e.g., wind, rain, avian calls) and mechanical threats (e.g., chainsaws, vehicles) with high precision. The system serves as a digital twin for a distributed IoT network, integrating data ingestion, model training, strategy simulation, and a centralized command dashboard.

## System Architecture

The solution is architected as a modular pipeline with four distinct stages:

1.  **Automated ETL Pipeline**: A robust data ingestion module that handles the download, extraction, and structural organization of the ESC-50 dataset. It enforces idempotency to ensure consistent data states across different environments.
2.  **Deep Learning Core**: Utilizes a ResNet-18 architecture modified for single-channel spectral input. The pipeline converts raw audio waveforms into Mel-Spectrograms, treating acoustic classification as a computer vision task.
3.  **Strategy Simulation Engine**: A headless simulation script designed to evaluate different alert threshold strategies (Aggressive vs. Conservative). This module mimics a production environment, processing randomized data streams to calculate False Positive Rates (FPR) and False Negative Rates (FNR).
4.  **Operational Command Center**: A Streamlit-based visualization layer. It provides geospatial localization of threats, displays real-time telemetry from simulated sensor nodes (heartbeat, battery, latency), and renders spectral analysis for human-in-the-loop verification.

## Repository Structure

The project follows a standard source-layout pattern to ensure maintainability and separation of concerns.

```text
ECOEAR/
├── assets/                  # Static resources for documentation and UI
├── data/                    # Local dataset storage (Managed by ETL pipeline)
│   ├── danger/              # Positive samples (Anthropogenic threats)
│   └── safe/                # Negative samples (Ambient biophony)
├── models/                  # Serialized model artifacts
│   └── ecoear_model.pth     # PyTorch state dictionary
├── src/                     # Application source code
│   ├── app.py               # Dashboard entry point
│   ├── data_loader.py       # ETL and preprocessing script
│   ├── simulation.py        # A/B testing and QA simulation
│   └── train.py             # Training loop and validation logic
├── .gitignore               # Version control exclusion rules
├── requirements.txt         # Python dependency manifest
└── README.md                # Project documentation
```

---

## Setup and Reproduction
This project is designed for full reproducibility. Follow the sequence below to initialize the environment and execute the pipeline.

1. Environment Initialization
Ensure Python 3.8+ is installed. It is recommended to use a virtual environment.
```Bash
git clone [https://github.com/YOUR_USERNAME/EcoEar.git](https://github.com/YOUR_USERNAME/EcoEar.git)
cd EcoEar
pip install -r requirements.txt
```

2. Data Ingestion (ETL)
Execute the data loader to pull the raw dataset and organize it into the training structure. This script handles network retries and file verification.
```Bash
python src/data_loader.py
```

3. Model Training
Initiate the training loop. The script uses Weighted Random Sampling to address class imbalance and saves the optimal model weights to the models/ directory based on validation accuracy.
```Bash
python src/train.py
```

4. Simulation & QA
Before deployment, run the simulation script to verify model inference logic and evaluate performance metrics on a randomized stream.
```Bash
python src/simulation.py
```

5. Deployment
Launch the interactive command center. This will start a local server and open the dashboard in your default web browser.
```Bash
streamlit run src/app.py
```
---

## Technical Specifications
- Input Data: Raw WAV audio sampled at 22,050 Hz.

- Feature Engineering: Log-Mel Spectrograms (128 Mel bands, FFT window=2048, Hop length=512).

- Model Backbone: ResNet-18 (Pretrained on ImageNet), fine-tuned with a modified input layer for 3-channel spectral compositing.

- Inference Latency: Optimized for <30ms per sample on MPS (Metal Performance Shaders) or CUDA enabled devices.

- Telemetry Simulation: Mocks a network of 50 distributed IoT nodes with realistic heartbeat logs and network latency metrics.

---

**Author:** Tanghao Chen (Dios) **Institution:** Carnegie Mellon University
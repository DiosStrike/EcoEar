<div align="center">

# [English](#english) | [中文](#chinese)

</div>

<span id="english"></span>


# EcoEar: Distributed Acoustic Monitoring System for Forest Conservation

### ** Live Demo :** [https://ecoear-demo.streamlit.app/](https://ecoear-demo.streamlit.app/) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecoear-demo.streamlit.app/)

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

## **1. Environment Initialization**
Ensure Python 3.8+ is installed. It is recommended to use a virtual environment.
```Bash
git clone [https://github.com/YOUR_USERNAME/EcoEar.git](https://github.com/YOUR_USERNAME/EcoEar.git)
cd EcoEar
pip install -r requirements.txt
```

## **2. Data Ingestion (ETL)**
Execute the data loader to pull the raw dataset and organize it into the training structure. This script handles network retries and file verification.
```Bash
python src/data_loader.py
```

## **3. Model Training**
Initiate the training loop. The script uses Weighted Random Sampling to address class imbalance and saves the optimal model weights to the models/ directory based on validation accuracy.
```Bash
python src/train.py
```

## **4. Simulation & QA (A/B Testing)**
Execute the headless simulation engine to perform **Sensitivity Analysis**. The script runs a comparative A/B test between "Aggressive" (Threshold 0.6) and "Conservative" (Threshold 0.85) strategies, generating a detailed report on False Positive/Negative Rates (FPR/FNR).
```bash
python src/simulation.py
```

## **5. Deployment**
Launch the interactive command center. This will start a local server and open the dashboard in your default web browser.
```Bash
streamlit run src/app.py
```
---

## Technical Specifications
- **Input Data**: Raw WAV audio sampled at 22,050 Hz.

- **Feature Engineering**: Log-Mel Spectrograms (128 Mel bands, FFT window=2048, Hop length=512).

- **Model Backbone**: ResNet-18 (Pretrained on ImageNet), fine-tuned with a modified input layer for 3-channel spectral compositing.

- **Inference Latency**: Optimized for <30ms per sample on MPS (Metal Performance Shaders) or CUDA enabled devices.

- **Telemetry Simulation**: Mocks a network of 50 distributed IoT nodes with realistic heartbeat logs and network latency metrics.

- **Strategy Evaluation**: Dual-stream A/B testing framework to quantify the trade-off between Recall (Security) and Precision (False Alarm Rate) across randomized stochastic data streams.

---

**Author:** Tanghao Chen (Dios) **Institution:** Carnegie Mellon University

---

</div>

<span id="chinese"></span>

# EcoEar: 分布式森林声学生态监测系统

### ** 在线演示 :** [https://ecoear-demo.streamlit.app/](https://ecoear-demo.streamlit.app/) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecoear-demo.streamlit.app/)

![System Banner](assets/forest.jpg)

## 摘要

非法伐木和人为侵占对全球森林生态系统构成了严重威胁。传统的卫星监测往往存在延迟，难以在实时环境下防止破坏。**EcoEar** 通过建立一个可扩展的声学传感器网络仿真系统，填补了这一空白。

本项目实现了一个端到端的机器学习工程 (MLE) 全流程管道，旨在对环境音频流进行分类。它能以高精度区分**自然生物声**（如风声、雨声、鸟叫）与**机械威胁**（如电锯、车辆）。该系统作为分布式物联网 (IoT) 网络的**数字孪生 (Digital Twin)**，集成了数据摄取、模型训练、策略仿真以及集中式指挥仪表盘。

## 系统架构

该解决方案架构为一个包含四个独立阶段的模块化管道：

1.  **自动化 ETL 管道**: 一个鲁棒的数据摄取模块，负责处理 ESC-50 数据集的下载、提取和结构化组织。它强制执行**幂等性 (Idempotency)**，以确保不同环境下数据状态的一致性。
2.  **深度学习核心**: 采用针对单通道频谱输入进行修改的 ResNet-18 架构。管道将原始音频波形转换为**梅尔频谱图 (Mel-Spectrograms)**，将声学分类视为计算机视觉任务处理。
3.  **策略仿真引擎**: 一个无头 (Headless) 仿真脚本，用于评估不同的报警阈值策略（激进型 vs. 保守型）。该模块模拟生产环境，处理随机化数据流以计算**误报率 (FPR)** 和 **漏报率 (FNR)**。
4.  **运营指挥中心**: 基于 Streamlit 的可视化层。它提供威胁的地理空间定位，显示来自模拟传感器节点的实时遥测数据（心跳、电池、延迟），并渲染频谱分析以供**人在回路 (Human-in-the-loop)** 验证。

## 仓库结构

本项目遵循标准的源码布局模式，以确保可维护性与关注点分离。

```text
ECOEAR/
├── assets/                  # 静态资源 (用于文档和 UI)
├── data/                    # 本地数据集存储 (由 ETL 管道管理)
│   ├── danger/              # 正样本 (人为威胁，如电锯声)
│   └── safe/                # 负样本 (环境生物声)
├── models/                  # 模型序列化产物
│   └── ecoear_model.pth     # PyTorch 状态字典 (State Dictionary)
├── src/                     # 应用程序源代码
│   ├── app.py               # 仪表盘入口点
│   ├── data_loader.py       # ETL 与预处理脚本
│   ├── simulation.py        # A/B 测试与 QA 仿真
│   └── train.py             # 训练循环与验证逻辑
├── .gitignore               # 版本控制排除规则
├── requirements.txt         # Python 依赖清单
└── README.md                # 项目文档
```

## 设置与复现 
本项目旨在实现完全的可复现性。请按照以下顺序初始化环境并执行管道。

## **1. 环境初始化**

确保已安装 Python 3.8+。建议使用虚拟环境。
```Bash
git clone [https://github.com/YOUR_USERNAME/EcoEar.git](https://github.com/YOUR_USERNAME/EcoEar.git)
cd EcoEar
pip install -r requirements.txt
```

## **2. 数据摄取 (ETL)**

执行数据加载器以拉取原始数据集并将其组织为训练结构。该脚本包含网络重试机制和文件校验。
```Bash
python src/data_loader.py
```

## **3. 模型训练**

启动训练循环。该脚本使用加权随机采样 (Weighted Random Sampling) 来解决类别不平衡问题，并根据验证准确率将最优模型权重保存到 models/ 目录。
```Bash
python src/train.py
```

## **4. 仿真与 QA (A/B 测试)**

执行无头仿真引擎进行 敏感性分析 (Sensitivity Analysis)。该脚本在“激进型”（阈值 0.6）和“保守型”（阈值 0.85）策略之间运行对比 A/B 测试，生成关于误报率/漏报率 (FPR/FNR) 的详细报告。
```Bash
python src/simulation.py
```

5. 部署

启动交互式指挥中心。这将启动一个本地服务器并在默认浏览器中打开仪表盘。
```Bash
streamlit run src/app.py
```

## 技术规格
- **输入数据**: 采样率为 22,050 Hz 的原始 WAV 音频。

- **特征工程**: 对数梅尔频谱图 (Log-Mel Spectrograms, 128 Mel bands, FFT window=2048, Hop length=512)。

- **模型骨干**: ResNet-18 (ImageNet 预训练)，微调修改后的输入层以适应 3 通道频谱合成。

- **推理延迟**: 针对 MPS (Metal Performance Shaders) 或 CUDA 设备进行了优化，单样本处理 <30ms。

- **遥测仿真**: 模拟包含 50 个分布式 IoT 节点的网络，具有逼真的心跳日志和网络延迟指标。

- **策略评估**: 双流 A/B 测试框架，用于量化随机数据流中召回率（安全性）与精确率（误报率）之间的权衡。

---

作者: 陈唐昊 | 机构: 卡耐基梅隆大学

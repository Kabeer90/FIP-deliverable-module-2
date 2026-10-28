# AI & Machine Learning Internship Deliverables

This repository contains two production-grade machine learning projects showcasing Deep Learning (Computer Vision) and Reinforcement Learning (Deep Q-Networks), complete with training pipelines and interactive Streamlit web applications.

---

## 📁 Repository Overview

```text
├── project1_deep_learning/
│   ├── dl_app.py              # Interactive Streamlit digit canvas application
│   ├── dl_training.ipynb      # PyTorch CNN training pipeline on MNIST
│   └── cnn_model.pth          # Pre-trained CNN weights (98.5%+ test accuracy)
├── project2_reinforcement_learning/
│   ├── rl_app.py              # Streamlit CartPole simulation & GIF generator
│   ├── rl_training.ipynb      # PyTorch DQN training pipeline with Replay Buffer
│   └── dqn_agent.pth          # Pre-trained DQN agent weights (solved CartPole-v1)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🧠 Project 1: Automated Score Digitizer (Deep Learning)

### 📌 Summary
An end-to-end computer vision solution for digit recognition. A custom Convolutional Neural Network (CNN) is trained on the MNIST handwritten digit dataset and deployed as a real-time web application where users can draw digits on an interactive canvas.

### 🔬 Architecture & Highlights
- **Model**: `DigitCNN` with two convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool2D) followed by a 3-layer MLP with Dropout.
- **Accuracy**: Achieves **>98.5%** accuracy on the MNIST test set.
- **Web App**: Built with **Streamlit** and `streamlit-drawable-canvas`, preprocessing user-drawn strokes into 28x28 normalized tensors, predicting digits with full softmax confidence bar charts.

### 🚀 Running Project 1
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit web app
streamlit run project1_deep_learning/dl_app.py
```

---

## 🕹️ Project 2: DQN Physics Optimizer (Reinforcement Learning)

### 📌 Summary
A Reinforcement Learning agent trained using **Deep Q-Networks (DQN)** to solve the classic control problem `CartPole-v1` from Gymnasium. The agent balances a pole upright on a moving cart by applying discrete horizontal forces.

### 🔬 Architecture & Highlights
- **Algorithm**: Deep Q-Learning with:
  - **Experience Replay Buffer** (10,000 transition capacity) for sample decorrelation.
  - **Target Network** updated every 10 episodes for training stability.
  - **$\epsilon$-Greedy Policy** with exponential decay ($1.0 \to 0.01$).
- **Solved Benchmark**: Consistently balances for the maximum duration (500 steps), exceeding the benchmark threshold of 475 average reward.
- **Web App**: Interactive Streamlit dashboard enabling users to compare the trained DQN agent against a random baseline, visualize telemetry plots (pole angles, positions, velocities), and render high-fps animated GIFs.

### 🚀 Running Project 2
```bash
# Run the Streamlit web app
streamlit run project2_reinforcement_learning/rl_app.py
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+ installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/Kabeer90/<repo-name>.git
cd <repo-name>
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Summary Table

| Feature | Project 1: Score Digitizer | Project 2: Physics Optimizer |
| :--- | :--- | :--- |
| **Domain** | Deep Learning / Computer Vision | Reinforcement Learning / Control |
| **Framework** | PyTorch & Torchvision | PyTorch & Gymnasium |
| **Algorithm** | Custom CNN + CrossEntropyLoss | Deep Q-Network (DQN) + Huber Loss |
| **Dataset/Env** | MNIST (60,000 train / 10,000 test) | Gymnasium `CartPole-v1` |
| **UI Interface** | Streamlit + Canvas Drawing | Streamlit + Dynamic GIF Simulation |

---

## 👤 Author
- **Kabir Abubakar** ([@Kabeer90](https://github.com/Kabeer90))

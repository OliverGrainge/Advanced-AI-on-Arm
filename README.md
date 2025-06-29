# 🧪 Advanced AI on Arm

This course provides a hands-on introduction to *extreme model quantization*, *hardware-aware optimization*, and *on-device deployment* for generative AI models. You'll explore advanced techniques to reduce model size, accelerate inference, and deploy compact LLMs on edge devices like Android smartphones.

## 🧬 Labs Overview

### 🔹 Lab 1: **Extreme Quantization**
Train a language model and progressively quantize it from FP32 to 8-bit, 4-bit, 2-bit, and 1-bit precision. Implement and evaluate **Quantization-Aware Training (QAT)** to mitigate accuracy degradation in ultra-low-bit models.

### 🔹 Lab 2: **Hardware–Software Model Co-Design**
Wrap all `nn.Linear` layers with a custom `QLinear` module and explore **layerwise post-training quantization**. Search for the optimal bit-width configuration to maximize efficiency while maintaining model fidelity in a software-hardware co-design process.

### 🔹 Lab 3: **Running & Quantizing Models on Android**
Use [`llama.cpp`](https://github.com/ggerganov/llama.cpp) to quantize and deploy LLaMA-style LLMs on Android. Learn how to benchmark and run models *offline*, directly on your mobile hardware.

---

## 🚀 Getting Started

This repository uses a unified `requirements.txt` and Git LFS to manage dependencies and large pretrained models.

### 1️⃣ Clone the Repository and Download Model Weights

```bash
# Install Git LFS if needed
sudo apt install git-lfs              # or: brew install git-lfs
git lfs install

# Clone the repo and pull large files
git clone https://github.com/your-org/Advanced-AI-on-Arm.git
cd Advanced-AI-on-Arm
git lfs pull
```

### 2️⃣ Set Up the Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Run the Labs

```bash
jupyter lab
```

Open:

- `lab1.ipynb` for **Extreme Quantization**
- `lab2.ipynb` for **Hardware–Software Co-Design**
- Follow `lab3.md` for **Android deployment** with `llama.cpp`

---

## 📁 Repository Structure

```
Advanced-AI-on-Arm/
├── lab1.ipynb/
├── lab2.ipynb/
├── lab3.md/
├── src/ 
├── assets/
├── requirements.txt
└── README.md
```

---

## 📱 Android Deployment Notes

To complete **Lab 3**, make sure the following are installed:

- Android Studio (Hedgehog or later)
- Android NDK + ADB
- A physical Android 10+ device with ≥6GB RAM

> Windows users: use **WSL 2** with Ubuntu 22.04 for full compatibility with build tools.

---

## 🧠 Learning Outcomes

- Understand bit-width trade-offs (accuracy vs. compression)
- Apply QAT to recover performance in quantized models
- Perform per-layer hardware-aware optimization
- Deploy and benchmark local LLMs on Android devices

---

## 📫 Questions?

Open an issue or contact `oliver@grainge.me` if you encounter problems during setup or execution.

---

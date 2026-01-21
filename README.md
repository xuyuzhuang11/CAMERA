# 🍀CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis

<div align="center">
  <a href="Paper">
    <img src="https://img.shields.io/badge/AAAI-2026-b31b1b.svg" alt="Arxiv Paper Link">
  </a>
  <a href="License">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="GitHub Code License">
  </a>
  <a href="https://github.com/xuyuzhuang11/CAMERA">
    <img src="https://img.shields.io/github/v/release/xuyuzhuang11/CAMERA" alt="GitHub release version">
  </a>
  <a href="Python">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version">
  </a>
  <h3>
    <a href="./README.md">Standard Version</a>
    &nbsp; | &nbsp;
    <a href="./Ascend/README.md">Ascend Version</a>
  </h3>
</div>

<p align="center"><strong>Yuzhuang Xu</strong>, Xu Han, Yuanchi Zhang, Yixuan Wang, Yijun Liu, Shiyu Ji, Qingfu Zhu, Wanxiang Che*</p>
<p align="center">Harbin Institute of Technology, Tsinghua University, WeChat</p>
<p align="center">In proceedings of <strong>AAAI 2026</strong></p>

<div align="center">
  <br>
  <img src="github.png" style="width: 25%">
</div>

<p align="center">
  <br>
  <strong>🔥 The First Micro-expert-based Structured Pruning Method 🔥</strong>
  <br>
  <strong>😱 Multi-matrix Joint Compression for Better Performance 😱</strong>
  <br>
  <strong>🤗 Welcome to Discuss with Us and Dig Out More Interesting Results 🤗</strong>
  <br>
</p>

---

## 📖 Introduction

Large Language Models (LLMs) with Mixture-of-Experts (MoE) architectures are distinguished by their strong performance scaling with increasing parameters across a wide range of tasks. However, they also suffer from substantial computational and storage overheads. Notably, the performance gains of MoE models do not scale proportionally with the growth in expert parameters.

While prior works attempt to reduce parameters via expert-level pruning, merging, or decomposition, they often encounter challenges in both performance preservation and computational efficiency. **CAMERA** addresses these challenges by introducing the concept of the "**micro-expert**", a finer-grained compression unit that spans across matrices.

This repository contains the implementation of **CAMERA**, a lightweight and training-free framework designed to identify and exploit micro-expert redundancy. Our approach allows for significant model compression while maintaining high performance, offering a fundamental shift in how MoE layers are analyzed and optimized.

### ✨ Core Features

* **🤖 Multi-matrix Joint Compression**: CAMERA is the first **micro-expert**-based compression method.
* **✂️ Structured Pruning**: Accurately identify redundant weights and obtain efficient structured pruning results.
* **📉 Good Performance**: Our method achieves excellent results compared to strong baselines.
* **🚀 Extremely Fast Speed**: It can compress the models with hundreds of billions of weights in a matter of minutes.

Please refer to our 📃[paper](https://arxiv.org/pdf/2508.02322) for more details.

---

## 🚀 Method Overview

The core of our methodology relies on the re-interpretation of MoE layers as mixtures of micro-experts, rather than just coarse-grained experts. A micro-expert is defined as a compression unit that spans across the weight matrices within the Feed-Forward Networks (FFN) of the MoE layers.

The CAMERA framework consists of three main components:

1. **CAMERA (Redundancy Analysis)**:

    - A lightweight, training-free algorithm that accurately ranks micro-experts based on their importance.

    - Our analysis reveals significant variance in micro-expert contributions during the decoding process, enabling the identification of redundant components without expensive retraining.

    - Efficiency: The method is highly efficient, capable of performing a complete micro-expert analysis of the Qwen2-57B-A14B model in less than 5 minutes on a single NVIDIA A100-40GB GPU.

2. **CAMERA-P (Structured Pruning)**:

    - A structured pruning framework that leverages the redundancy analysis to remove less important micro-experts.

    - It prunes jointly across the matrices in each FFN, preserving the functional integrity and coordination of the remaining parameters.

3. **CAMERA-Q (Mixed-Precision Quantization)**:

    - A novel micro-expert-aware quantization strategy.

    - Allocates bit-widths dynamically based on micro-expert importance, allowing for aggressive quantization (e.g., 2-bit) of redundant micro-experts while maintaining higher precision for critical ones.

---

## 📊 Experimental Results

We conducted extensive experiments to validate the effectiveness of CAMERA across various settings:

- **Benchmarks**: Evaluated on nine downstream tasks, covering a diverse range of capabilities including reasoning and language understanding.

- **Pruning Performance (CAMERA-P)**:

    - Consistently outperforms strong baselines (including expert-level pruning and merging methods) across pruning ratios ranging from 20% to 60%.

    - Demonstrates superior retention of model capabilities even at high compression rates.

- **Quantization Performance (CAMERA-Q)**:

    - Achieves superior results under aggressive 2-bit quantization regimes.

    - Surpasses existing matrix-level and channel-level quantization techniques by effectively identifying which parameters can tolerate lower precision.

- **Computational Efficiency**:

    - The analysis and compression process is orders of magnitude faster than existing methods, which often require hours of computation on multi-GPU setups.

### DeepSeek-16b Pruning Performance

The 20% structured pruning results are as follows:

| Model | Wiki2 | C4 | Wino | Hella | BoolQ | ARC-e | MathQA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek-MoE-16b-base** | 6.51 | 9.05 | 70.24 | 77.37 | 72.45 | 72.89 | 31.52 |
| NAEE | 6.77 | 10.07 | 69.53 | 74.63 | 67.83 | **72.56** | 30.99 |
| D2-MoE | 7.29 | 12.62 | 69.22 | 69.87 | **69.32** | 71.29 | 29.45 |
| **CAMERA** | **6.57** | **9.84** | **70.17** | **75.02** | 68.01 | 71.80 | **31.46** |

> **Analysis**: CAMERA achieves the fastest running of all baselines, taking only 3 minutes. In contrast, NAEE takes 12 hours and D2-MoE takes 3 hours.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/xuyuzhuang11/CAMERA.git
cd CAMERA

# 2. Create environment
conda create -n camera python=3.10
conda activate camera

```

The main classes and functions are defined in mei.py. The auxiliary files are datautils.py and modelutils.py.

The model specific implementations are in d16b.py, qwen2.py, qwen3.py and mixtral.py.

---

## 💻 Usage

### 1. Data & Model Preparation

Download the base models and Wiki2/C4 dataset.

```bash
# Example: Download DeepSeek-moe-16b-base
git lfs clone [git clone https://huggingface.co/deepseek-ai/deepseek-moe-16b-base](git clone https://huggingface.co/deepseek-ai/deepseek-moe-16b-base)

```

### 2. Run CAMERA-P

Use the main script to load the model and apply CAMERA optimization.

```bash
bash run.sh

```


### 3. Evaluation

Evaluate the pruned model on PPL and tasks:

```bash
bash run_eval.sh

```

---

## 📝 Citation

If you use CAMERA in your research, please cite our work:

```bibtex
@inproceedings{xu2025camera,
  title={CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis},
  author={Xu, Yuzhuang and Han, Xu and Zhang, Yuanchi and Wang, Yixuan and Liu, Yijun and Ji, Shiyu and Zhu, Qingfu and Che, Wanxiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026},
  url={https://arxiv.org/pdf/2508.02322}
}
```

---

## 📧 Contact

For questions regarding the code or paper, please contact:

* **Yuzhuang Xu**: [xyz@ir.hit.edu.cn](mailto:xyz@ir.hit.edu.cn)

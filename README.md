
# Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining

**Official implementation of the paper:**
 *Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining*
Accepted at **AICCSA 2025**

# Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining

📄 [Read the Paper on arXiv](https://arxiv.org/abs/2508.15828)


![hippo](https://github.com/sazzadadib/Z-Pruner/blob/main/Arc%20Design.gif)
<figcaption style="color: red; font-size: smaller;">This figure illustrates the overall method of our pruning technique</figcaption>

---

## Overview

**Z-Pruner** is a post-training pruning framework designed for large language models (LLMs). Unlike traditional pruning methods, **Z-Pruner**:

* Requires **no retraining or fine-tuning** after pruning.
* Maintains **competitive perplexity** and inference efficiency.
* Supports major transformer-based models like **LLaMA** and **OPT**.

This repository provides:

* Source code for pruning large-scale transformer-based LLMs.
* Example scripts for applying Z-Pruner on LLaMA and OPT models.
* Evaluation code for measuring **perplexity** and **zero-shot accuracy**.
* Baselines for comparison.

---

## Repository Structure

```
Z-Pruner/
├── LICENSE                 # License file
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── main.py                 # Entry point for running pruning/evaluation
└── lib/                    # Core implementation
    ├── data.py             # Dataset loading and preprocessing
    ├── eval.py             # Perplexity and efficiency evaluation
    ├── layerwrapper.py     # Transformer layer wrappers
    ├── prune.py            # Z-Pruner core algorithm
    ├── quant.py            # Optional quantization routines
```

---

## Installation

```bash
git clone https://github.com/sazzadadib/Z-Pruner.git
cd Z-Pruner
pip install -r requirements.txt
```

---

##  Evaluation

### 1. Perplexity Evaluation

```
python main.py \
    --model YOUR_MODEL_NAME \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
```

> **Note:** For LLaMA models, set your HuggingFace token in `main.py`.

---

### 2. Zero-Shot Accuracy

Follow the installation guide for the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to measure zero-shot accuracy.

---

## Results


### Perplexity on Wikitext2 (50% Unstructured Sparsity)

| **Method**          | **OPT 1.3B** | **OPT 2.7B** | **OPT 6.7B** | **LLaMA-2 7B** | **LLaMA-2 13B** | **LLaMA-3.1 8B** |
| ------------------- | ------------ | ------------ | ------------ | -------------- | --------------- | ---------------- |
| Wanda               | 18.41        | 14.22        | 15.21        | 7.76           | 6.29            | 11.53            |
| SparseGPT           | **17.55**    | **13.46**    | 11.62        | 7.01           | 6.03            | 9.86             |
| RIA                 | 18.08        | 14.20        | 11.83        | 6.81           | 5.83            | 9.44             |
| **Z-Pruner (Ours)** | 17.74        | 13.92        | **11.60**    | **6.74**       | **5.82**        | **9.37**         |

> **Bold values** indicate the best (lowest) perplexity among all methods.

---

### Zero-Shot Accuracy (LLaMA-2-7B, 50% Sparsity)

| **Method**          | **HellaSwag** | **BoolQ** | **WinoGrande** | **MNLI**  | **WNLI**  | **Average** | **Pruning Time (min)** |
| ------------------- | ------------- | --------- | -------------- | --------- | --------- | ----------- | ---------------------- |
| Magnitude           | 49.13         | 63.00     | 63.30          | 31.57     | 38.45     | 49.09       | **4.51**               |
| SparseGPT           | 52.75         | **76.48** | **69.30**      | 38.57     | 40.85     | 55.59       | 35.15                  |
| Wanda               | 50.32         | 75.05     | 67.80          | 38.14     | 42.25     | 54.71       | 13.47                  |
| RIA                 | 52.04         | 74.22     | 68.27          | 39.31     | 42.25     | 55.22       | 13.52                  |
| **Z-Pruner (Ours)** | **52.79**     | 74.98     | 68.51          | **39.40** | **43.66** | **55.87**   | 11.81                  |

> **Bold values** indicate the best (highest) accuracy or lowest pruning time across tasks.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{bhuiyan2025zprunerposttrainingpruninglarge,
      title={Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining}, 
      author={Samiul Basir Bhuiyan and Md. Sazzad Hossain Adib and Mohammed Aman Bhuiyan and Muhammad Rafsan Kabir and Moshiur Farazi and Shafin Rahman and Nabeel Mohammed},
      year={2025},
      eprint={2508.15828},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.15828}, 
}
```


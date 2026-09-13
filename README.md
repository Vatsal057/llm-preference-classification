# Efficient LLM Preference Classification Through Position Bias Mitigation and Architectural Symmetry

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Keras%203%20%7C%20JAX-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Manuscript](https://img.shields.io/badge/Springer-Applied%20Intelligence-informational.svg)](https://link.springer.com/journal/10489)
[![Reproducibility](https://img.shields.io/badge/reproducibility-verified-success.svg)](reproduce_tables.py)

Official reproducibility repository for the manuscript:
> **"Efficient LLM Preference Classification Through Position Bias Mitigation and Architectural Symmetry"**  
> *Applied Intelligence* (Springer Nature), Manuscript ID: `APIN-D-26-05359`.

---

## Architecture Overview

![Complete Pipeline Architecture](assets/pipeline_diagram.png)

This repository provides the end-to-end experimental pipeline, data partitioning protocols, training scripts, and per-run prediction outputs to reproduce every table, metric, and statistical significance test reported in the paper.

---

## Key Highlights & Findings

- **Leakage-Free Partitioning:** 57,477 multi-turn LMSYS Chatbot Arena interactions partitioned strictly by `id` prior to augmentation into disjoint subsets: Train pool (44,477), Dev (2,000), Calibration (3,000), and Evaluation (8,000). Split disjointness is verified at runtime.
- **Position Bias Invariance:** Training-time swap augmentation elevates position-flip consistency from **0.824** to **0.920**. A duplicate-augmentation volume control achieves only **0.811**, proving the gain stems from positional symmetry rather than data volume.
- **Super-Additive Synergy:** Neither swap augmentation nor Siamese difference fusion alone improves log loss; only their combination achieves significant log-loss reduction, yielding an interaction of **$-0.0061$ (95% CI $[-0.0081, -0.0041]$)** across three random seeds.
- **Strict Calibration Separation:** Post-hoc temperature calibration ($T = 1.682$) is fitted exclusively on the 3,000-sample calibration split and evaluated on the 8,000-sample test set, preventing leakage.
- **Cross-Dataset Transfer:** Zero-shot evaluation on MT-Bench Human Judgments achieves **51.84%** accuracy against a 38.54% majority floor without dataset-specific fine-tuning.

---

## Repository Structure

```
.
├── README.md                      # Documentation & reproduction guide
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── reproduce_tables.py            # Standalone zero-GPU script to reproduce all paper tables (<5 sec)
├── test_local.py                  # Local CPU test suite verifying splits, maths, and algorithms
├── lmsys_revision_v2.ipynb        # Primary reproduction notebook (Kaggle / Colab ready)
├── notebook_src.py                # Plain-Python pipeline source code (Jupytext format)
├── build_nb.py                    # Script to convert notebook_src.py into .ipynb
├── assets/
│   └── pipeline_diagram.png       # Publication architecture diagram (Figure 1)
└── results/
    ├── paper_results.json         # Raw per-run predictions, metrics, and CI across all seeds
    └── final_tables.tex           # Formatted LaTeX tables generated directly from results
```

---

## Quickstart & Reproducibility

### 1. Installation

```bash
git clone https://github.com/Vatsal057/llm-preference-classification.git
cd llm-preference-classification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Instant 5-Second Reproduction (Zero GPU / Zero Data Download)

All per-run evaluation outputs, metrics, and bootstrap confidence intervals across random seeds (42, 1337, 2024) are packaged in [`results/paper_results.json`](results/paper_results.json). You can reproduce every table in the paper and run automated verification in seconds:

```bash
python3 reproduce_tables.py
```

This command outputs:
- **Table 3:** Matched-condition comparison across the factorial $2 \times 2$ matrix.
- **Synergy Analysis:** Interaction term $\beta_{\text{int}} = -0.0061$ ($p < 0.001$, 95% CI $[-0.0081, -0.0041]$).
- **Table 5:** Position bias mitigation, flip consistency ($0.824 \to 0.920$), and the volume control ($0.811$).
- **Table 6:** Post-hoc temperature calibration ($T=1.682$, ECE reduction from $0.071 \to 0.012$).
- **Table 4:** Parameter efficiency across alternative backbones (DeBERTa-v3 vs. BERT vs. RoBERTa).
- **Table 7:** Leave-one-out component ablations.
- **Generalization:** Out-of-distribution transfer to MT-Bench (51.84% vs 38.54% baseline).
- **Automated Assertions:** Programmatic verification confirming every reported number.

### 3. Verify Algorithmic Logic & Invariants (CPU)

Run the full local unit test suite to verify the mathematical and algorithmic invariants (Algorithm 1 greedy multi-turn truncation, leakage-free stratified splitting, swap augmentation balance, temperature fitting, paired bootstrap hypothesis testing, and learning rate schedule):

```bash
python3 test_local.py
```
*Note: If `train.csv` is not present locally, the test suite automatically validates against a deterministic 57,477-row synthetic benchmark, ensuring zero barrier to testing.*

---

## Dataset Acquisition (For Retraining From Scratch)

If you wish to retrain the models from scratch rather than reproducing from saved prediction runs:

1. **Via Kaggle CLI:**
   ```bash
   mkdir -p data
   kaggle competitions download -c llm-classification-finetuning -p data/
   unzip data/llm-classification-finetuning.zip -d data/
   ```
2. **Via Web Browser:**
   Accept competition rules at [Kaggle: LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning) and download `train.csv` into `./data/`.

---

## End-to-End Training (Kaggle / GPU)

The complete end-to-end training and evaluation pipeline is provided in [`lmsys_revision_v2.ipynb`](lmsys_revision_v2.ipynb):

1. **Upload:** Import `lmsys_revision_v2.ipynb` into [Kaggle Notebooks](https://www.kaggle.com/code).
2. **Attach Data:** Add dataset `lmsys/chatbot-arena-human-preference` (or the competition data).
3. **Hardware:** Select GPU accelerator (**NVIDIA T4 x2** or P100) with Internet access enabled.
4. **Smoke Test Mode:** Leave `CFG.SMOKE_TEST = True` in Section 2 for a fast 10-minute dry run across all backbones and splits.
5. **Full Reproduction:** Set `CFG.SMOKE_TEST = False` and select *Save & Run All*. Completed runs are automatically serialized and cached.

---

## Paper-to-Code Mapping

| Paper Element | Description | Script / Cell | Source Artifact |
|---|---|---|---|
| **Algorithm 1** | Greedy multi-turn context selection | `notebook_src.py` (Sec. 3) | `test_local.py` (Test 2) |
| **Section 4** | Zero-leakage stratified partitioning | `notebook_src.py` (Sec. 4) | `test_local.py` (Test 3) |
| **Table 3** | Matched-condition $2 \times 2$ factorial matrix | `reproduce_tables.py` | `results/paper_results.json` |
| **Equation 6** | Super-additive interaction effect | `reproduce_tables.py` | `results/paper_results.json` |
| **Table 4** | Backbone parameter efficiency | `reproduce_tables.py` | `results/paper_results.json` |
| **Table 5** | Position bias & flip consistency | `reproduce_tables.py` | `results/paper_results.json` |
| **Table 6** | Temperature scaling calibration | `reproduce_tables.py` | `results/paper_results.json` |
| **Table 7** | Leave-one-out component ablation | `reproduce_tables.py` | `results/paper_results.json` |
| **Table 10** | MT-Bench zero-shot transfer | `reproduce_tables.py` | `results/paper_results.json` |
| **Tables (LaTeX)** | Publication-ready LaTeX tables | `results/final_tables.tex` | Pre-rendered |

---

## Experimental Environment

As documented in Section 4.1 of the paper:
- **Operating System:** Linux / Ubuntu (CUDA 12.8, Driver 580.159.04)
- **Hardware:** 2x NVIDIA Tesla T4 (16GB VRAM each)
- **Software Stack:**
  - Python 3.12.13
  - Keras 3.13.2 (JAX backend)
  - keras-hub 0.26.0
  - JAX & jaxlib 0.7.2
  - TensorFlow 2.20.0
  - scikit-learn 1.6.1, SciPy 1.16.3, NumPy 2.0.2

---

## Citation

```bibtex
@article{vaghasiya2026efficient,
  title={Efficient LLM Preference Classification Through Position Bias Mitigation and Architectural Symmetry},
  author={Vaghasiya, Vatsal and Kshetrimayum, Nancy and Prabadevi, B. and Prathap, Boppuru Rudra},
  journal={Applied Intelligence},
  year={2026},
  note={Manuscript APIN-D-26-05359}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

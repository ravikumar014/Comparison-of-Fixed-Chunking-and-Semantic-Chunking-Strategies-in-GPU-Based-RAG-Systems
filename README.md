# Comparison of Fixed Chunking and Semantic Chunking Strategies in GPU-Based RAG Systems

## Overview

This project presents a comparative study of **Fixed Chunking** and **Semantic Chunking** strategies in Retrieval-Augmented Generation (RAG) systems.

The goal is to investigate how different chunking approaches affect:

* Retrieval Quality
* Ranking Performance
* Semantic Coherence
* Downstream RAG Generation

The system implements an end-to-end GPU-compatible RAG pipeline consisting of:

1. PDF Document Processing
2. Text Cleaning
3. Fixed Chunking
4. Semantic Chunking
5. Sentence Embeddings
6. FAISS Vector Retrieval
7. RAG-based Answer Generation
8. Retrieval Evaluation and Ablation Studies

---

## Research Objective

The primary research question is:

> Does semantic chunking improve retrieval quality compared to traditional fixed chunking in dense vector retrieval systems?

Evaluation is performed using:

* Recall@k
* Precision@k
* Mean Reciprocal Rank (MRR)
* nDCG@k

---

## Project Structure

```text
Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── chunks/
│
├── artifacts/
│   ├── fixed_chunks.json
│   ├── semantic_chunks.json
│   ├── fixed_embeddings.npy
│   └── semantic_embeddings.npy
│
├── results/
│   ├── retrieval_metrics.json
│   ├── ablation_results.json
│   ├── rag_answers_fixed.json
│   ├── rag_answers_semantic.json
│   └── plots/
│
├── src/
│   ├── loaders/
│   ├── preprocessing/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── evaluation/
│   ├── rag/
│   └── utils/
│
├── experiments/
│   ├── stage1_data_stats.py
│   ├── stage2_fixed_chunking.py
│   ├── stage3_semantic_chunking.py
│   ├── stage4_embeddings_faiss.py
│   ├── stage5_retrieval_eval.py
│   ├── stage6_rag_generation.py
│   ├── ablation_chunk_size.py
│   ├── ablation_k.py
│   └── plot_results.py
│
├── README.md
└── THEORY_AND_MATH.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Project
```

### Create Virtual Environment

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Hardware Support

The project automatically detects available hardware.

Priority:

1. NVIDIA CUDA GPU
2. Apple Silicon MPS
3. CPU

Supported Platforms:

* NVIDIA CUDA
* Apple M-Series (MPS)
* CPU-only Systems

---

## Dataset Preparation

Place research papers inside:

```text
data/raw/
```

Example:

```text
data/raw/
├── paper1.pdf
├── paper2.pdf
├── paper3.pdf
└── ...
```

Recommended:

* 20–50 research papers
* Same domain/topic
* PDF format

---

## Execution Pipeline

### Stage 1 — Document Loading

```bash
python -m experiments.stage1_data_stats
```

Loads and analyzes PDF documents.

---

### Stage 2 — Fixed Chunking

```bash
python -m experiments.stage2_fixed_chunking
```

Generates fixed-size chunks.

---

### Stage 3 — Semantic Chunking

```bash
python -m experiments.stage3_semantic_chunking
```

Generates semantically coherent chunks.

---

### Stage 4 — Embeddings + FAISS

```bash
python -m experiments.stage4_fixed_embeddings_faiss
```

```bash
python -m experiments.stage4_semantic_embeddings_faiss
```

Creates:

* Embeddings
* FAISS Index
for both fixed and semantic.

---

### Stage 5 — Retrieval Evaluation

```bash
python -m experiments.stage5_retrieval_evaluation
```

Computes:

* Recall@k
* Precision@k
* MRR
* nDCG

---

### Stage 6 — RAG Generation

```bash
python -m experiments.stage6_rag_generation
```

Generates answers using retrieved context.

---

## Ablation Studies

### Chunk Size Ablation

```bash
python -m experiments.ablation_chunk_size
```

Tests:

* 256
* 512
* 768

chunk sizes.

---

### Top-k Ablation

```bash
python -m experiments.ablation_k
```

Tests retrieval performance for:

* k = 1
* k = 3
* k = 5
* k = 10

---

## Results

Example Findings:

| Method            | Recall@5                 | MRR                        |
| ----------------- | ------------------------ | -------------------------- |
| Fixed Chunking    | Higher baseline          | Lower ranking quality      |
| Semantic Chunking | Higher retrieval quality | Better ranking performance |

Key Observation:

> Semantic chunking consistently improves semantic coherence and retrieval ranking effectiveness.

---

## Technologies Used

* Python
* PyTorch
* SentenceTransformers
* Transformers
* FAISS
* NumPy
* Pandas
* Matplotlib
* PyMuPDF

---

## Future Work

Potential improvements:

* Hybrid Chunking
* Hierarchical Chunking
* Query-Aware Chunking
* Reranking Models
* Multimodal RAG
* Adaptive Semantic Thresholds

---

## Citation

If you use this work, please cite:

```bibtex
@article{fixed_vs_semantic_chunking,
  title={Comparison of Fixed Chunking and Semantic Chunking Strategies in GPU-Based RAG Systems},
  author={Ravi Kumar U},
  year={2026}
}
```

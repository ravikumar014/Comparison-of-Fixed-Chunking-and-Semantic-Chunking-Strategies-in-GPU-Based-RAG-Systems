# Theory and Mathematics Behind the Project

## Introduction

This project investigates the impact of chunking strategies on retrieval performance in Retrieval-Augmented Generation (RAG) systems.

The study compares:

1. Fixed Chunking
2. Semantic Chunking

using dense vector retrieval and transformer-based embeddings.

---

# Retrieval-Augmented Generation (RAG)

A RAG pipeline combines retrieval with generation:

```mermaid
flowchart LR
    Q[User Query] --> R[Retriever]
    R --> C[Relevant Chunks]
    C --> L[LLM]
    L --> A[Generated Answer]
```

The retrieval stage determines what information the language model receives.

Therefore:

```mermaid
flowchart LR
    A[Retrieval Quality] --> B[Answer Quality]
```

---

# Fixed Chunking

## Definition

A document is divided into equal-sized token windows.

Document:

$$
D = {w_1,w_2,\dots,w_n}
$$

Chunk:

$$
C_i
===

{w_{s_i},\dots,w_{s_i+k}}
$$

where:

* $k$ = chunk size
* $s_i$ = starting index

---

## Overlap

To preserve context:

$$
s_{i+1}
=======

s_i + (k-o)
$$

where:

* $o$ = overlap size

---

## Advantages

* Fast
* Deterministic
* Scalable

---

## Disadvantages

* Sentence fragmentation
* Context loss
* Mixed semantics

---

# Semantic Chunking

## Core Idea

Instead of splitting by token count, split according to semantic similarity.

---

## Sentence Embeddings

Sentence:

$$
s_i
$$

Embedding:

$$
e_i = f(s_i)
$$

where:

$$
e_i \in \mathbb{R}^{768}
$$

---

## Cosine Similarity

$$
\text{sim}(e_i,e_j)
===================

\frac{e_i \cdot e_j}
{|e_i||e_j|}
$$

Interpretation:

* 1 → highly similar
* 0 → unrelated
* -1 → opposite

---

## Boundary Rule

If:

$$
\text{sim}(e_i,e_{i+1}) < \tau
$$

then:

Create a new chunk.

---

# Transformer Attention

Sentence embeddings are generated using transformers.

Query:

$$
Q = XW_Q
$$

Key:

$$
K = XW_K
$$

Value:

$$
V = XW_V
$$

Self-Attention:

$$
\text{Attention}(Q,K,V)
=======================

\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

This mechanism captures contextual relationships between tokens.

---

# Embedding Space Geometry

Embeddings exist in high-dimensional vector space.

Semantically similar sentences satisfy:

$$
|x_i-x_j|
\rightarrow
0
$$

Semantically unrelated sentences satisfy:

$$
|x_i-x_j|
\gg
0
$$

Semantic chunking exploits this property.

---

# FAISS Retrieval

## Full Form

FAISS

=
Facebook AI Similarity Search

---

## Objective

Given query embedding:

$$
q
$$

retrieve:

$$
\arg\max_i
\text{cosine}(q,x_i)
$$

---

## Index Used

IndexFlatIP

where:

IP = Inner Product

For normalized vectors:

$$
q \cdot x_i
===========

\text{cosine similarity}
$$

---

# Retrieval Metrics

## Recall@k

$$
\text{Recall@k}
===============

\frac{\text{Relevant Retrieved}}
{\text{Total Relevant}}
$$

Measures retrieval coverage.

---

## Precision@k

$$
\text{Precision@k}
==================

\frac{\text{Relevant Retrieved}}
{k}
$$

Measures retrieval purity.

---

## Mean Reciprocal Rank (MRR)

$$
\text{MRR}
==========

\frac{1}{Q}
\sum_{i=1}^{Q}
\frac{1}{rank_i}
$$

Measures ranking quality.

---

## nDCG@k

$$
\text{nDCG@k}
=============

\frac{\text{DCG@k}}
{\text{IDCG@k}}
$$

Measures ranking effectiveness.

---

# Computational Complexity

## Fixed Chunking

$$
O(n)
$$

---

## Semantic Chunking

$$
O(n \cdot d)
$$

where:

* $n$ = sentences
* $d$ = embedding dimension

---

## FAISS Retrieval

Exact retrieval:

$$
O(Nd)
$$

where:

* $N$ = total embeddings
* $d$ = embedding dimension

---

# Key Research Finding

The central hypothesis of this project is:

$$
\boxed{
\text{Semantically coherent chunk boundaries improve dense retrieval effectiveness.}
}
$$

Experimental results demonstrate that semantic chunking improves:

* Recall@k
* MRR
* nDCG
* Answer Coherence

compared to traditional fixed chunking approaches.

---

# References

* RAG (Lewis et al., 2020)
* Sentence-BERT (Reimers and Gurevych, 2019)
* FAISS (Johnson et al., 2019)
* Transformer Architecture (Vaswani et al., 2017)

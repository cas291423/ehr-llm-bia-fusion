# ehr-llm-bia-fusion
An LLM-enhanced multimodal EHR representation framework
# LLM-Enhanced Semantic Multimodal Fusion with Bidirectional Attention for EHR Representation
This repository contains the complete implementation of our framework LLM-Enhanced Semantic Multimodal Fusion with Bidirectional Attention, designed for fine-grained representation learning and risk prediction from multimodal Electronic Health Records (EHRs).
The pipeline is divided into two main components:

Q3E.py: Embedding generation via a locally hosted LLM (Qwen3-Embedding-4B).
Structured EHR features are textualized and passed into the LLM to preserve clinical semantics.
Unstructured clinical narratives are segmented into fixed-length sequences (23 tokens), padded with [MASK] tokens when shorter.
Each token is embedded into a fixed 32-dimensional representation.

BiAttnFusionSeq.py: Multimodal fusion and prediction.
Intra-modal dependencies are modeled by self-attention layers.
Cross-modal alignment is performed using bidirectional attention ([inspired by BiDAF README.md](https://github.com/galsang/BiDAF-pytorch/blob/master/README.md#bidaf-pytorch)]).
A masked pooling layer ensures padded [MASK] tokens do not interfere with downstream computations.
The concatenated pooled representations are passed through a 3-layer MLP classifier with hidden sizes [256, 128, 64].

# Environment Setup
Dependencies
Python 3.11.13；
PyTorch  2.3.1；
scikit-learn 1.7.1；
pandas 2.3.1；numpy 1.26.4；
optuna 4.4.0；
gradio_client (for embedding API call)1.10.4

# Running the Embedding Server ( Qwen3-Embedding Installation Guide)
Installation Guide for Large Language Models Document: Qwen3-Embedding Installation Guide
 After installation, run:
python qwen3_embedding.py
You should see output like:
Running on local URL:  http://127.0.0.1:7860

# Step 1: Generate Embeddings (Q3E.py)
Input: sampled_data_share.xlsx (31 structured features + 1 text column + 1 label column).

Output: embedding_sampled_data_share.xlsx

For each structured column: a new {col}_emb column containing a 32-dimensional JSON vector.

For the text column: 23 sentence columns {text_col}_sent{i} each holding a 32-dimensional JSON vector or "MASK".

The label column is preserved unchanged.

why？？23 sentence columns--The sample with the most segmented text contains 23 text data entries.

# Step 2: Train Multimodal Fusion Model (BiAttnFusionSeq.py)

Dataset is split 8:1:1 into train/validation/test.

Self-attention and BiAttn modules are trained end-to-end.

A 3-layer MLP classifier with hidden sizes [256,128,64] produces predictions.

Validation PR-AUC (macro) is used as the selection criterion.

# Model Architecture

Semantic Serialization (TabTransformer-style tokens)

Structured indicators are textualized into [indicator_name, value, definition].

Clinical narratives are segmented into sentence tokens (t_1, …, t_n), normalized to 23 with [MASK] padding.

Each token → embedded into a 32-dimensional vector via Qwen3-Embedding-4B.

Inspired by TabTransformer [[Huang et al., NeurIPS 2020](https://github.com/lucidrains/tab-transformer-pytorch)], each structured/unstructured entry is treated as a semantic token.

Intra-modal Self-Attention

Bidirectional Cross-Modal Attention (BiDAF-inspired)

Masked Pooling & Fusion

Structured: mean pooling 

 (size = 256 when d_model=128).

Classifier (3-layer MLP)

Hidden dimensions: [256 → 128 → 64].

Output: number of target classes.

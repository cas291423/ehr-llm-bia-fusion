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
PyTorch >= 1.12；
scikit-learn；
pandas, numpy；
optuna；
gradio_client (for embedding API call)

# Running the Embedding Server
We follow https://www.bilibili.com/video/BV1gnMfzpE8M/?spm_id_from=333.1387.favlist.content.click&vd_source=49997804e3d2d06ca11e20d30162e95e
 to launch a local Gradio server exposing the Qwen3-Embedding-4B model.
 After installation, run:
python qwen3_embedding.py
You should see output like:
Running on local URL:  http://127.0.0.1:7860

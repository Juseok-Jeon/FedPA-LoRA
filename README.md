# FedPA-LoRA: Product-Aligned Federated LoRA

This repository provides the anonymized implementation accompanying the submitted paper on **FedPA-LoRA**.

FedPA-LoRA addresses two aggregation challenges in federated LoRA training: factor-level initialization mismatch across communication rounds and the discrepancy between factor-wise aggregation and the average of client LoRA products. Clients preserve their local LoRA factors and align their product updates with a low-rank global reference, while the server efficiently reconstructs the aggregated adapter without explicitly forming the dense update matrix.

## Overview

For client (i), the LoRA update is represented as

[
\Delta W_i = B_i A_i.
]

FedPA-LoRA preserves the local factors (B_i) and (A_i) across rounds and regularizes the client product toward a rank-constrained global reference. After local training, the server approximates the average product

[
\frac{1}{N}\sum_{i=1}^{N} B_i A_i
]

using reduced QR decomposition and a small core SVD.

## Highlights

* **Local factor preservation:** Retains client-specific LoRA factors across communication rounds.
* **Product alignment:** Aligns each local product (B_iA_i) with a low-rank global reference.
* **Product-aware aggregation:** Reconstructs the global adapter from client products rather than directly averaging mismatched factors.
* **Rank flexibility:** Supports heterogeneous local and communication ranks.
* **Efficient reconstruction:** Avoids explicitly forming the dense aggregated update matrix.

## Implementation

The implementation is built on top of FederatedScope-LLM. The main components are located in:

* `federatedscope/core/workers/client.py`: Client-side local training and factor preservation.
* `federatedscope/core/workers/server.py`: Server-side product aggregation and reconstruction.
* `federatedscope/core/aggregators/`: Aggregation utilities.
* `federatedscope/core/configs/cfg_llm.py`: LoRA and FedPA-LoRA configuration options.
* `federatedscope/glue/yamls/ours.yaml`: Configuration for GLUE experiments.
* `federatedscope/llm/yamls/ours.yaml`: Configuration for generative tasks.

## Installation

The code has been tested with Python 3.10 and PyTorch.

```shell
conda create -n fedpa-lora python=3.10
conda activate fedpa-lora

pip install -e .[llm]
pip install evaluate
```

Install the PyTorch version appropriate for your CUDA environment before installing this package.

## Quick Start

Run FedPA-LoRA on GLUE-style natural language understanding tasks:

```shell
python -m federatedscope.main \
  --cfg federatedscope/glue/yamls/ours.yaml
```

Run FedPA-LoRA on LLM generative tasks:

```shell
python -m federatedscope.main \
  --cfg federatedscope/llm/yamls/ours.yaml
```

## Anonymity Notice

This repository has been anonymized for double-blind peer review. Author names, affiliations, acknowledgements, citation information, and identifying links will be added after the review process.

## Acknowledgements

Acknowledgement information is omitted during double-blind review.

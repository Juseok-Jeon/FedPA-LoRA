# FedPA-LoRA: Product-Aligned Federated LoRA

This repository provides the anonymized implementation of **FedPA-LoRA**, a federated LoRA framework designed to address factor-level initialization mismatch and factor-wise aggregation mismatch.

FedPA-LoRA preserves each client's locally optimized LoRA factors across communication rounds, guides local training through product-level alignment with a low-rank global reference, and aggregates client updates directly in the product space.

## Overview

For client $i$, the LoRA update is represented as

$$
\Delta W_i = B_i A_i.
$$

Conventional federated LoRA methods typically replace local factors with newly aggregated global factors at every communication round. This may disrupt optimization continuity because different factor pairs can represent similar product updates.

FedPA-LoRA instead preserves each client's local factors across rounds:

$$
(B_i^{(t,0)}, A_i^{(t,0)})
\leftarrow
(B_i^{(t-1)}, A_i^{(t-1)}).
$$

To control client drift, each client minimizes a product-guided local objective:

$$
\mathcal{L}_i^{(t)}(B_i,A_i)
============================

f_i(W_0+B_iA_i)
+
\frac{\lambda}{2}
\left|
B_iA_i
------

B_{g,i}^{(t-1)}A_{g,i}^{(t-1)}
\right|_{\mathrm{F}}^2.
$$

Here, $W_0$ is the frozen pretrained weight matrix and
$(B_{g,i}^{(t-1)},A_{g,i}^{(t-1)})$ is a rank-$R_i$ global reference determined by the client's communication budget.

After local training, the server aggregates client updates directly in the product space:

$$
\Delta W_{\mathrm{ideal}}^{(t)}
===============================

\frac{1}{N}
\sum_{i=1}^{N}
B_i^{(t)}A_i^{(t)}.
$$

This avoids the mismatch caused by independently averaging $B_i$ and $A_i$.

## Highlights

* **Local factor preservation:** Retains locally optimized LoRA factors across communication rounds.
* **Product-guided alignment:** Regularizes each local product toward a low-rank global reference.
* **Product-space aggregation:** Aggregates $B_iA_i$ rather than averaging the two factors independently.
* **Heterogeneous ranks:** Supports different client computation ranks $r_i$ and communication ranks $R_i$.
* **Efficient reconstruction:** Reconstructs the global adapter using reduced QR decomposition and core SVD.
* **Randomized extension:** Supports a lower-complexity randomized reconstruction option.

## Efficient Global Reconstruction

The server constructs concatenated factors

$$
B_{\mathrm{cat}}^{(t)}
======================

\frac{1}{\sqrt{N}}
\left[
B_1^{(t)},\ldots,B_N^{(t)}
\right],
$$

$$
A_{\mathrm{cat}}^{(t)}
======================

\frac{1}{\sqrt{N}}
\left[
(A_1^{(t)})^\top,\ldots,(A_N^{(t)})^\top
\right]^\top.
$$

These factors satisfy

$$
B_{\mathrm{cat}}^{(t)}A_{\mathrm{cat}}^{(t)}
============================================

\Delta W_{\mathrm{ideal}}^{(t)}.
$$

The server applies reduced QR factorizations to the concatenated factors and performs truncated SVD only on the resulting small core matrix. This reconstructs the optimal rank-$R_g$ approximation without explicitly forming the dense aggregated update.

## Implementation

The implementation is built on top of FederatedScope-LLM. The main components are located in:

* `federatedscope/core/workers/client.py`: Local factor preservation and product-guided training.
* `federatedscope/core/workers/server.py`: Global reference construction and server-side reconstruction.
* `federatedscope/core/aggregators/`: Aggregation utilities.
* `federatedscope/core/configs/cfg_llm.py`: LoRA and FedPA-LoRA configuration options.
* `federatedscope/glue/yamls/ours.yaml`: Configuration for natural language understanding tasks.
* `federatedscope/llm/yamls/ours.yaml`: Configuration for generative tasks.

## Installation

The implementation uses Python 3.10 and PyTorch.

```shell
conda create -n fedpa-lora python=3.10
conda activate fedpa-lora

pip install -e .[llm]
pip install evaluate
```

Install the PyTorch build corresponding to your CUDA environment before installing this package.

## Quick Start

Run a natural language understanding experiment:

```shell
python -m federatedscope.main \
  --cfg federatedscope/glue/yamls/ours.yaml
```

Run a generative-task experiment:

```shell
python -m federatedscope.main \
  --cfg federatedscope/llm/yamls/ours.yaml
```

## Server-Side Complexity

Under homogeneous client rank $r$ and $Nr \ll d$, the exact reconstruction has dominant per-layer server complexity

$$
\mathcal{O}(N^2dr^2),
$$

while the randomized reconstruction reduces it to

$$
\mathcal{O}(Ndr^2).
$$

Both approaches avoid explicitly constructing the dense $d \times d$ product aggregate.

## Anonymity Notice

This repository has been anonymized for double-blind peer review. Author names, affiliations, acknowledgements, paper links, and citation information are intentionally omitted during the review period.
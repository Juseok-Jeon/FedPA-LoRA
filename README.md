# FedPA-LoRA: Product-Aligned Federated LoRA

This repository provides the anonymized implementation of **FedPA-LoRA**, a federated LoRA framework that addresses aggregation and initialization mismatches in federated LoRA training.

## Highlights

* **Product-space aggregation:** Aggregates client products $B_iA_i$ rather than averaging $B_i$ and $A_i$ independently.
* **Local factor preservation:** Retains locally optimized factors across rounds to avoid initialization mismatch.
* **Product-guided alignment:** Controls client drift using a low-rank global reference while supporting heterogeneous ranks.
* **Efficient reconstruction:** Reduces dense-SVD complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(N^2dr^2)$ using reduced QR, and further to $\mathcal{O}(Ndr^2)$ using randomized reconstruction.

## Overview

For client $i$, the LoRA update is represented as $\Delta W_i=B_iA_i$.

FedPA-LoRA preserves the previously optimized local factors across communication rounds instead of replacing them with newly reconstructed global factors. Local products are guided toward rank-$R_i$ global references to mitigate client drift without disrupting the local optimization trajectory.

After local training, the server directly targets the product-space aggregate

```math
\Delta W_{\mathrm{ideal}}^{(t)}
=
\frac{1}{N}\sum_{i=1}^{N}B_i^{(t)}A_i^{(t)}.
```

This avoids factor-wise aggregation mismatch and naturally supports heterogeneous client ranks.

## Efficient Reconstruction

Directly forming the dense product aggregate and applying SVD requires dominant server-side complexity $\mathcal{O}(d^3)$. FedPA-LoRA instead applies reduced QR factorizations to concatenated low-rank factors and performs truncated SVD only on the resulting small core matrix, reducing the complexity to $\mathcal{O}(N^2dr^2)$ while recovering the same optimal rank-constrained approximation.

A factored randomized variant further reduces the complexity to $\mathcal{O}(Ndr^2)$ when the sketch dimension is $\mathcal{O}(r)$, at the cost of a probabilistic approximation guarantee.

## Installation

```shell
conda create -n fedpa-lora python=3.10
conda activate fedpa-lora
pip install -e .[llm]
pip install evaluate
```

Install the PyTorch build corresponding to your CUDA environment before installing this package.

## Implementation

FedPA-LoRA is implemented on top of FederatedScope-LLM. A representative experiment can be launched with

```shell
python -m federatedscope.main --cfg federatedscope/glue/yamls/ours.yaml
```

Additional configurations are provided under `federatedscope/glue/yamls/` and `federatedscope/llm/yamls/`.

## Anonymity Notice

This repository has been anonymized for double-blind peer review. Author names, affiliations, paper links, acknowledgements, and citation information are intentionally omitted during the review period.
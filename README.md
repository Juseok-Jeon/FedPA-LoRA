# FedPA-LoRA: Product-Aligned Federated LoRA

This repository provides the anonymized implementation of **FedPA-LoRA**, a federated LoRA framework that addresses two fundamental mismatches in federated LoRA training: aggregation mismatch from independently averaging LoRA factors and initialization mismatch from replacing locally optimized factors across communication rounds.

FedPA-LoRA resolves aggregation mismatch through product-space aggregation and low-rank reconstruction, while resolving initialization mismatch through local factor preservation.

## Highlights

* **Mismatch correction:** Product-space reconstruction resolves aggregation mismatch, while local factor preservation resolves initialization mismatch.
* **Product-guided training:** Controls client drift without overwriting locally optimized factors.
* **Rank flexibility:** Supports heterogeneous computation and communication ranks.
* **Efficient reconstruction:** Reduces dense-SVD complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(N^2dr^2)$ using reduced QR, and further to $\mathcal{O}(Ndr^2)$ using randomized reconstruction.

## Overview

For client $i$, the LoRA update is represented as

```math
\Delta W_i = B_iA_i.
```

### Local Factor Preservation

Instead of replacing local factors with newly reconstructed global factors at every communication round, each client preserves its previously optimized factors:

```math
(B_i^{(t,0)},A_i^{(t,0)})
\leftarrow
(B_i^{(t-1)},A_i^{(t-1)}).
```

This avoids factor-level initialization mismatch and maintains optimization continuity across rounds.

### Product-Guided Alignment

To control client drift without overwriting the preserved factors, client $i$ minimizes

```math
\mathcal{L}_i^{(t)}(B_i,A_i)
=
f_i(W_0+B_iA_i)
+
\frac{\lambda}{2}
\left\|
B_iA_i
-
B_{g,i}^{(t-1)}A_{g,i}^{(t-1)}
\right\|_{\mathrm F}^{2}.
```

Here, $W_0$ is the frozen pretrained weight matrix, and $(B_{g,i}^{(t-1)},A_{g,i}^{(t-1)})$ is a rank-$R_i$ global reference determined by the client's communication budget.

### Product-Space Aggregation

Rather than averaging $B_i$ and $A_i$ independently, the server aggregates client updates directly in the product space:

```math
\Delta W_{\mathrm{ideal}}^{(t)}
=
\frac{1}{N}
\sum_{i=1}^{N}
B_i^{(t)}A_i^{(t)}.
```

This resolves factor-wise aggregation mismatch and naturally supports heterogeneous client ranks.

## Efficient Reconstruction

Directly constructing the dense product aggregate and applying SVD requires dominant server-side complexity $\mathcal{O}(d^3)$.

FedPA-LoRA instead concatenates the uploaded low-rank factors such that

```math
B_{\mathrm{cat}}^{(t)}A_{\mathrm{cat}}^{(t)}
=
\frac{1}{N}
\sum_{i=1}^{N}
B_i^{(t)}A_i^{(t)}.
```

Reduced QR factorizations followed by truncated SVD of the resulting small core matrix recover the same optimal rank-$R_g$ approximation with dominant per-layer complexity

```math
\mathcal{O}(N^2dr^2).
```

A factored randomized reconstruction further reduces the dominant complexity to

```math
\mathcal{O}(Ndr^2)
```

when the sketch dimension is $\mathcal{O}(r)$, at the cost of replacing exact rank-constrained optimality with a probabilistic approximation guarantee.

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

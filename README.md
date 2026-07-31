# FedPA-LoRA: Product-Aligned Federated LoRA

This repository provides the anonymized implementation of **FedPA-LoRA**, a federated LoRA framework that addresses factor-level initialization mismatch and factor-wise aggregation mismatch.

FedPA-LoRA preserves each client's locally optimized LoRA factors across communication rounds, aligns local products with low-rank global references, and aggregates client updates directly in the product space.

## Overview

For client $i$, the LoRA update is represented as

```math
\Delta W_i = B_iA_i.
```

Instead of replacing the local factors with newly reconstructed global factors at every round, FedPA-LoRA preserves them as

```math
(B_i^{(t,0)},A_i^{(t,0)})
\leftarrow
(B_i^{(t-1)},A_i^{(t-1)}).
```

To control client drift while retaining local optimization continuity, client $i$ minimizes

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

After local training, the server targets the product-space aggregate

```math
\Delta W_{\mathrm{ideal}}^{(t)}
=
\frac{1}{N}
\sum_{i=1}^{N}
B_i^{(t)}A_i^{(t)}.
```

This avoids the mismatch caused by independently averaging the two LoRA factors.

## Highlights

* **Local preservation and product alignment:** Preserves client-specific factors while controlling drift in the product space.
* **Product-space aggregation:** Avoids factor-wise aggregation mismatch.
* **Rank flexibility:** Supports heterogeneous computation and communication ranks.
* **Efficient reconstruction:** Provides exact reduced-QR reconstruction and a lower-complexity randomized approximation.

## Efficient Reconstruction

The server constructs concatenated factors satisfying

```math
B_{\mathrm{cat}}^{(t)}A_{\mathrm{cat}}^{(t)}
=
\frac{1}{N}
\sum_{i=1}^{N}
B_i^{(t)}A_i^{(t)}.
```

The exact reconstruction applies reduced QR factorizations to the concatenated factors and truncated SVD to the resulting small core matrix. It recovers the optimal rank-$R_g$ approximation without explicitly forming the dense aggregate.

Under homogeneous rank $r$ and $Nr\ll d$, its dominant per-layer server complexity is

```math
\mathcal{O}(N^2dr^2).
```

The randomized reconstruction directly sketches the factored aggregate and reduces the dominant complexity to

```math
\mathcal{O}(Ndr^2)
```

when the sketch dimension is $\mathcal{O}(r)$, at the cost of replacing exact rank-constrained optimality with a probabilistic approximation guarantee.

## Implementation

The implementation is built on FederatedScope-LLM. The main components are located in:

* `federatedscope/core/workers/client.py`: Local factor preservation and product-guided training.
* `federatedscope/core/workers/server.py`: Global reference construction and server-side reconstruction.
* `federatedscope/core/aggregators/`: Aggregation utilities.
* `federatedscope/core/configs/cfg_llm.py`: LoRA and FedPA-LoRA configuration options.
* `federatedscope/glue/yamls/ours.yaml`: Natural language understanding configuration.
* `federatedscope/llm/yamls/ours.yaml`: Generative-task configuration.

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

Natural language understanding:

```shell
python -m federatedscope.main --cfg federatedscope/glue/yamls/ours.yaml
```

Generative tasks:

```shell
python -m federatedscope.main --cfg federatedscope/llm/yamls/ours.yaml
```

## Anonymity Notice

This repository has been anonymized for double-blind peer review. Author names, affiliations, paper links, acknowledgements, and citation information are intentionally omitted during the review period.
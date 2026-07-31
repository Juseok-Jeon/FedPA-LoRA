# FedPA-LoRA: Product-Aligned Federated LoRA

This repository provides the anonymized implementation of **FedPA-LoRA**, a federated LoRA framework that addresses aggregation and initialization mismatches in federated LoRA training.

## Highlights

* **Product-space aggregation:** Aggregates client products $B_iA_i$ rather than averaging $B_i$ and $A_i$ independently.
* **Local factor preservation:** Retains locally optimized factors across rounds to avoid initialization mismatch.
* **Product-guided alignment:** Controls client drift using a low-rank global reference while supporting heterogeneous ranks.
* **Efficient reconstruction:** Reduces dense-SVD complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(N^2dr^2)$ using reduced QR, and further to $\mathcal{O}(Ndr^2)$ using randomized reconstruction.

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
python -m federatedscope.main --cfg federatedscope/glue/yamls/fedpa.yaml
```

Additional configurations are provided under `federatedscope/glue/yamls/` and `federatedscope/llm/yamls/`.

## Anonymity Notice

This repository has been anonymized for double-blind peer review. Author names, affiliations, paper links, acknowledgements, and citation information are intentionally omitted during the review period.
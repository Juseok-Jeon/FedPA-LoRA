# FedPA-LoRA: Product-Aligned Federated LoRA

This repository provides the anonymized implementation of **FedPA-LoRA**, a federated LoRA framework that addresses aggregation and initialization mismatches in federated LoRA training.

## Highlights

* **Product-space aggregation:** Aggregates client products $B_iA_i$ rather than averaging $B_i$ and $A_i$ independently.
* **Local factor preservation:** Retains locally optimized factors across rounds.
* **Product-guided alignment:** Controls client drift while supporting heterogeneous ranks.
* **Efficient reconstruction:** Reduces server complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(N^2dr^2)$, or $\mathcal{O}(Ndr^2)$ with randomized reconstruction.

## Installation

```shell
conda create -n fedpa-lora python=3.10
conda activate fedpa-lora

# Install the PyTorch build compatible with your CUDA environment first.
pip install -e ".[llm]"
pip install evaluate
```

## Running an Experiment

```shell
python -m federatedscope.main --cfg federatedscope/glue/yamls/fedpa.yaml
```

Additional configurations are provided under `federatedscope/glue/yamls/` and `federatedscope/llm/yamls/`.

Reported results are averaged over seeds `0`, `13`, and `123`. The corresponding dataset, model, training, and method configurations are specified in the YAML files.

## Anonymity Notice

This repository has been anonymized for double-blind peer review. Author names, affiliations, paper links, acknowledgements, and citation information are intentionally omitted during the review period.
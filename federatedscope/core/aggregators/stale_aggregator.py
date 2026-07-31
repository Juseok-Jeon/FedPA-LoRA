import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
import copy
import numpy as np


def weight_minus(weights_A, weights_B):
    # get gradient by subtracting weights_next_round from weights_this_round
    # Ensure both are on the same device, usually CPU for aggregation
    weight_diff = {}
    for name in weights_A:
        wa = weights_A[name].cpu()
        wb = weights_B[name].cpu()
        weight_diff[name] = wa - wb
    return weight_diff

def flatten_weights(weights):
    """Flattens a state_dict into a single 1D tensor."""
    # Sort keys to ensure deterministic order
    return torch.cat([weights[k].reshape(-1) for k in sorted(weights.keys())])

def unflatten_weights(flat_tensor, template):
    """Unflattens a 1D tensor back into a state_dict matching the template."""
    new_weights = {}
    pointer = 0
    for k in sorted(template.keys()):
        shape = template[k].shape
        numel = template[k].numel()
        new_weights[k] = flat_tensor[pointer:pointer + numel].view(shape)
        pointer += numel
    return new_weights


class StaleAggregator(Aggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config
        self.s_maintain = None
        self.client_num = self.cfg.federate.client_num
        p_high = 0.16
        p_low = 0.04
        self.sampling_probs = np.array([p_high] * self.client_num, dtype=np.float32)
        self.sampling_probs[self.client_num // 2:] = p_low

    def update(self, model_parameters):
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def aggregate(self, agg_info, stale_weights, last_round_model):
        """
        Aggregates updates using FedSteer (Projection) or FedStale/FedVARP (Rescaling).

        Args:
            agg_info: Contains active client feedback and indices.
            stale_weights: Dict {client_id: weight_dict} of last known weights.
            last_round_model: State dict of the global model at the start of this round.
        """
        # 1. Setup & Unpack Info

        di = torch.zeros(self.client_num, dtype=torch.float32).cpu()
        all_client_ids = sorted(list(stale_weights.keys()))
        for client_id in all_client_ids:
            # Assuming client_id is an integer
            di[client_id - 1] = 1.0  # data samples num are mostly the same
        # 2. Normalize
        # torch.sum returns a tensor, so we use item() or let broadcasting handle it
        if di.sum() > 0:
            di = di / di.sum()
        # force di to be long type
        di = di.long()

        active_client_data = agg_info["client_feedback"]  # List of (sample_size, model_para)
        active_indices = agg_info["clients_idx"]  # List of client IDs (e.g. [1, 5, 10])

        # Prepare template for aggregation (copy of last round model)
        # We will modify this to become the new global model
        new_global_model = copy.deepcopy(last_round_model)
        # Initialize new_global_model to zeros if we are accumulating gradients from scratch
        # OR keep it as is if we are applying updates.
        # The logic in federated_stale effectively does: w_new = w_old - sum(gradients)
        # So we start with w_old (last_round_model) and subtract gradients.

        # 2. Calculate Fresh Gradients for Active Clients
        # g_i = w_global - w_local
        active_gradients = {}
        active_sample_sizes = {}

        # Build map of client_id -> gradient
        for idx, client_id in enumerate(active_indices):
            sample_size, client_model = active_client_data[idx]
            active_sample_sizes[client_id] = sample_size
            # Compute gradient
            active_gradients[client_id] = weight_minus(last_round_model, client_model)

        # 3. Prepare information for all clients (active & inactive)

        # Calculate/Estimate total data size for weighting
        # In FS, we often only know active sample sizes. We might need to track global sizes.
        # For now, assume simple averaging or use what's available.
        # If using probabilities (p_list), that logic needs to be passed in or config-based.

        # --- ALGORITHM SWITCH ---

        # Logic for MIFA
        if self.cfg.FedSteer.method == 'mifa':  # Check config key
            # MIFA Logic: Average all most recent gradients (fresh for active, stale for inactive)

            accumulated_grad = {
                k: torch.zeros_like(v)
                for k, v in last_round_model.items()
            }

            for cid in all_client_ids:
                # Determine gradient and weight for this client
                if cid in active_indices:
                    grad = active_gradients[cid]
                else:
                    # Stale gradient: g_stale = w_old_global - w_stale
                    # Note: stale_weights[cid] stores the weights sent by client last time they were active
                    # We compare it against the CURRENT global model to get the "gradient direction" relative to now?
                    # Actually MIFA usually keeps the *gradient* itself.
                    # If stale_weights stores weights, we need the global model *at that time* to get the true gradient.
                    # If simplified, we just use current global - stale weights, which is FedAvg with stale weights.
                    # Assuming stale_weights[cid] IS the weights:
                    grad = weight_minus(last_round_model, stale_weights[cid])

                # Accumulate
                for k in accumulated_grad:
                    accumulated_grad[k] += di[cid-1] * grad[k]

            # Apply update: w_new = w_old - eta * avg_grad
            # If inputs are weights, this is equivalent to averaging weights if eta=1
            for k in new_global_model:
                new_global_model[k] -= accumulated_grad[k]

        # Logic for FedSteer / FedStale
        else:
            # This block implements the general stale aggregation logic
            # w_new = w_old - sum( weight * corrected_gradient )

            accumulated_grad = {k: torch.zeros_like(v) for k, v in last_round_model.items()}

            # --- FedSteer Specifics (Optional placement) ---
            # If FedSteer, you would calculate 's' and 'Q' here using active_gradients
            # and stale_weights, similar to your `updateV` functions.
            # updated_stale_grads = updateV(...)
            # -----------------------------------------------

            for cid in all_client_ids:
                if cid in active_indices:
                    # Active client contribution
                    grad = active_gradients[cid]
                    # Weighting: d_i / p_i.
                    # p_i logic needs to be derived from config or passed in.
                    # For standard avg:
                    weight = active_sample_sizes[cid]

                    # If FedSteer, we might add a correction term here

                else:
                    # Inactive client contribution
                    # Calculate stale gradient
                    grad = weight_minus(last_round_model, stale_weights[cid])
                    weight = di[cid-1]

                    # Apply Beta (FedStale) attenuation if configured
                    if hasattr(self.cfg.FedSteer, 'beta') and self.cfg.FedSteer.beta > 0:
                        for k in grad:
                            grad[k] *= self.cfg.FedSteer.beta

                    # If FedSteer, we would project 'grad' using 's' and 'Q' here

                # Accumulate
                # Note: The denominator (normalization) depends on the specific algorithm's bounds.
                # Standard FedAvg divides by total_weight.
                for k in accumulated_grad:
                    accumulated_grad[k] += grad[k] * weight  # Simplified weighting

            # Normalize and Update
            # This normalization needs to be aligned with your p_i / d_i logic
            normalization = sum(active_sample_sizes.values())  # Simplified placeholder

            for k in new_global_model:
                if normalization > 0:
                    new_global_model[k] -= accumulated_grad[k] / normalization

        return new_global_model

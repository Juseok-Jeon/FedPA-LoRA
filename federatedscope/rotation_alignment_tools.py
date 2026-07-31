import torch
import logging
from collections import OrderedDict

from federatedscope.core.auxiliaries.utils import get_ds_rank

logger = logging.getLogger(__name__)
if get_ds_rank() == 0:
    logger.setLevel(logging.INFO)

def rotation_align_optimization(initial_model_ref: torch.Tensor,
                                align_matrix: str,
                                updated_A: torch.Tensor,
                                updated_B: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Performs rotation alignment for a *single pair* of LoRA parameters using PyTorch.

    This function solves the Orthogonal Procrustes problem to find the
    optimal rotation matrix (R) that aligns a client's updated parameters
    to a shared reference (the initial_model_ref). It then applies this
    rotation to both A and B.

    Args:
        initial_model_ref (torch.Tensor): The reference matrix (on the same device as updated_A/B).
        align_matrix (str): Either 'A' or 'B'. Specifies which matrix to use
            for the alignment calculation.
        updated_A (torch.Tensor): The client's locally updated A matrix (r x d).
        updated_B (torch.Tensor): The client's locally updated B matrix (d x r).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - rotated_A (torch.Tensor): The new A matrix (R.T @ A).
            - rotated_B (torch.Tensor): The new B matrix (B @ R).
    """

    # Get original dtype and device to ensure consistency
    # We'll use the dtypes of A and B directly for casting.
    a_dtype = updated_A.dtype
    b_dtype = updated_B.dtype
    original_device = updated_A.device

    # 1. Find the optimal rotation R by solving the Orthogonal Procrustes problem.
    # We want to find an r x r rotation matrix R.
    # All computations will stay on the device of the input tensors (e.g., 'cuda:0').

    with torch.no_grad():  # Ensure no gradients are computed during alignment
        if align_matrix == 'A':
            # M = updated_A @ initial_model_ref.T
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_A.to(torch.float32), initial_model_ref.T.to(torch.float32))
        elif align_matrix == 'B':
            # M = updated_B.T @ initial_model_ref
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_B.T.to(torch.float32), initial_model_ref.to(torch.float32))
        else:
            raise ValueError("align_matrix must be 'A' or 'B'")

        # 2. Compute SVD of the r x r correlation matrix M using torch.linalg.svd
        try:
            # --- FIX for 'Half' precision error ---
            # SVD on CUDA does not support float16/bfloat16. M is already float32.
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except torch.linalg.LinAlgError:
            print(
                f"Warning: SVD computation failed for matrix of shape {M.shape} on device {M.device}. Using identity matrix.")
            # Fallback to a simple identity matrix on the same device
            R_32 = torch.eye(M.shape[0], device=original_device, dtype=torch.float32)
        else:
            # The optimal rotation matrix R = U @ Vh
            # R is computed in float32.
            R_32 = torch.matmul(U, Vh)

        # 3. Apply the rotation to both A and B
        # B' = B @ R
        # A' = R.T @ A

        # --- FIX for dtype mismatch ---
        # Explicitly cast R (which is float32) to the *specific* dtype
        # of the matrix it is being multiplied with.

        # Cast R to B's dtype for the first multiplication
        # rotated_B = torch.matmul(updated_B, R_32.to(b_dtype))
        # Cast R.T to A's dtype for the second multiplication
        # rotated_A = torch.matmul(R_32.T.to(a_dtype), updated_A)
            
        rotated_B = torch.matmul(
            updated_B.float(),
            R_32.to(device=updated_B.device, dtype=torch.float32)
        ).to(dtype=b_dtype)
        rotated_A = torch.matmul(
            R_32.T.to(device=updated_A.device, dtype=torch.float32),
            updated_A.float()
        ).to(dtype=a_dtype)

    return rotated_A, rotated_B


def rotation_align_optimization_soft(initial_model_ref: torch.Tensor,
                                align_matrix: str,
                                updated_A: torch.Tensor,
                                updated_B: torch.Tensor,
                                rotation_lambda) -> (torch.Tensor, torch.Tensor):
    """
    Performs rotation alignment for a *single pair* of LoRA parameters using PyTorch.

    This function solves the Orthogonal Procrustes problem to find the
    optimal rotation matrix (R) that aligns a client's updated parameters
    to a shared reference (the initial_model_ref). It then applies this
    rotation to both A and B.

    Args:
        initial_model_ref (torch.Tensor): The reference matrix (on the same device as updated_A/B).
        align_matrix (str): Either 'A' or 'B'. Specifies which matrix to use
            for the alignment calculation.
        updated_A (torch.Tensor): The client's locally updated A matrix (r x d).
        updated_B (torch.Tensor): The client's locally updated B matrix (d x r).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - rotated_A (torch.Tensor): The new A matrix (R.T @ A).
            - rotated_B (torch.Tensor): The new B matrix (B @ R).
    """

    # Get original dtype and device to ensure consistency
    # We'll use the dtypes of A and B directly for casting.
    a_dtype = updated_A.dtype
    b_dtype = updated_B.dtype
    original_device = updated_A.device

    # 1. Find the optimal rotation R by solving the Orthogonal Procrustes problem.
    # We want to find an r x r rotation matrix R.
    # All computations will stay on the device of the input tensors (e.g., 'cuda:0').

    with torch.no_grad():  # Ensure no gradients are computed during alignment
        if align_matrix == 'A':
            # M = updated_A @ initial_model_ref.T
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_A.to(torch.float32), initial_model_ref.T.to(torch.float32))
        elif align_matrix == 'B':
            # M = updated_B.T @ initial_model_ref
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_B.T.to(torch.float32), initial_model_ref.to(torch.float32))
        else:
            raise ValueError("align_matrix must be 'A' or 'B'")

        # 2. Compute SVD of the r x r correlation matrix M using torch.linalg.svd
        try:
            # --- FIX for 'Half' precision error ---
            # SVD on CUDA does not support float16/bfloat16. M is already float32.
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except torch.linalg.LinAlgError:
            print(
                f"Warning: SVD computation failed for matrix of shape {M.shape} on device {M.device}. Using identity matrix.")
            # Fallback to a simple identity matrix on the same device
            R_32 = torch.eye(M.shape[0], device=original_device, dtype=torch.float32)
        else:
            # The optimal rotation matrix R = U @ Vh
            # R is computed in float32.
            R_32 = torch.matmul(U, Vh)

        # If determinant is -1, SVD gave us a reflection (mirror), not a rotation.
        # This causes catastrophic cancellation. We must flip it back.
        if torch.linalg.det(R_32) < 0:
            # Flip the sign of the last row of Vh (or last col of U)
            Vh[-1, :] *= -1
            R_32 = torch.matmul(U, Vh)
            print('need to flip for rotation matrix')
        if rotation_lambda == 1.0:
            pass
        else:
            # Linear Interpolation: (1 - lambda) * I + lambda * R_target
            I = torch.eye(R_32.shape[0], device=original_device, dtype=torch.float32)
            R_soft = (1 - rotation_lambda) * I + rotation_lambda * R_32

            # Re-orthogonalize: Project R_soft back to the closest orthogonal matrix
            # We do this by taking the SVD of the interpolated matrix: R_soft = U' S' V'T
            # The closest orthogonal matrix is U' V'T
            try:
                U_s, _, Vh_s = torch.linalg.svd(R_soft, full_matrices=False)
                R_32 = torch.matmul(U_s, Vh_s)
            except torch.linalg.LinAlgError:
                # Fallback to hard rotation if re-orthogenalization fails
                pass

        # Cast R to B's dtype for the first multiplication
        # rotated_B = torch.matmul(updated_B, R_32.to(b_dtype))
        # Cast R.T to A's dtype for the second multiplication
        # rotated_A = torch.matmul(R_32.T.to(a_dtype), updated_A)
            
        rotated_B = torch.matmul(
            updated_B.float(),
            R_32.to(device=updated_B.device, dtype=torch.float32)
        ).to(dtype=b_dtype)
        rotated_A = torch.matmul(
            R_32.T.to(device=updated_A.device, dtype=torch.float32),
            updated_A.float()
        ).to(dtype=a_dtype)

    return rotated_A, rotated_B





def rotation_alignment(initial_model_ref: dict,
                       align: str,
                       updated_A: dict,
                       updated_B: dict) -> (dict, dict):
    """
    Wrapper function to align an entire LoRA model (represented as dictionaries).

    This iterates through layers and calls the torch-based optimization.

    Args:
        initial_model_ref (dict): The reference model weights (e.g., A_ref_dict or B_ref_dict).
        align (str): Either 'A' or 'B'.
        updated_A (dict): Client's updated A matrices.
        updated_B (dict): Client's updated B matrices.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - rotatedA (dict): The new, aligned A matrices.
            - rotatedB (dict): The new, aligned B matrices.
    """
    rotatedA = OrderedDict()
    rotatedB = OrderedDict()
    layer_keys_A = list(updated_A.keys())
    # Create corresponding B keys (e.g., 'layer1.lora_A' -> 'layer1.lora_B')
    layer_keys_B = [key.replace('.lora_A', '.lora_B') for key in layer_keys_A]  # ensure order

    if align == 'A':
        ref_keys = layer_keys_A
    elif align == 'B':
        ref_keys = layer_keys_B
    else:
        raise ValueError("align must be 'A' or 'B'")

    for key_a, key_b, key_ref in zip(layer_keys_A, layer_keys_B, ref_keys):
        # Check if the keys exist before processing
        if key_a not in updated_A or key_b not in updated_B or key_ref not in initial_model_ref:
            print(f"Warning: Skipping layer due to missing keys. Looked for {key_a}, {key_b}, {key_ref}")
            continue

        A = updated_A[key_a]
        B = updated_B[key_b]
        model_ref = initial_model_ref[key_ref]

        # Call the pure PyTorch function
        rotatedA[key_a], rotatedB[key_b] = rotation_align_optimization(model_ref, align, A, B)

    return rotatedA, rotatedB


def random_rotation_matrix(r: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Sample a Haar-uniform random rotation matrix in SO(r).
    Uses QR of a Gaussian matrix and fixes the sign / determinant.
    """
    # Gaussian matrix
    Z = torch.randn((r, r), device=device, dtype=dtype)

    # QR decomposition
    # Q is orthonormal, R is upper-triangular
    Q, R = torch.linalg.qr(Z)

    # Make Q uniform by forcing diag(R) to be positive
    # (this removes bias from QR sign ambiguity)
    diag = torch.diagonal(R)
    sign = torch.sign(diag)
    sign[sign == 0] = 1.0
    Q = Q * sign  # broadcast on columns

    # Enforce det(Q)=+1 (SO(r)) rather than a reflection
    if torch.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    return Q


def rotation_align_random(updated_A: torch.Tensor, updated_B: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Apply a random rotation (SO(r)) to a single LoRA pair (A, B).
    Preserves BA exactly: (B R)(R^T A) = BA.
    """
    a_dtype = updated_A.dtype
    b_dtype = updated_B.dtype
    device = updated_A.device
    r = updated_A.shape[0]

    with torch.no_grad():
        R_32 = random_rotation_matrix(r, device=device, dtype=torch.float32)

        # rotated_B = torch.matmul(updated_B, R_32.to(b_dtype))
        # rotated_A = torch.matmul(R_32.T.to(a_dtype), updated_A)

        rotated_B = torch.matmul(
            updated_B.float(),
            R_32.to(device=updated_B.device, dtype=torch.float32)
        ).to(dtype=b_dtype)
        rotated_A = torch.matmul(
            R_32.T.to(device=updated_A.device, dtype=torch.float32),
            updated_A.float()
        ).to(dtype=a_dtype)

    return rotated_A, rotated_B


def random_rotation_function(updated_A: dict, updated_B: dict) -> (dict, dict):
    # soft rotation alignment
    rotatedA = OrderedDict()
    rotatedB = OrderedDict()
    layer_keys_A = list(updated_A.keys())
    # Create corresponding B keys (e.g., 'layer1.lora_A' -> 'layer1.lora_B')
    layer_keys_B = [key.replace('.lora_A', '.lora_B') for key in layer_keys_A]  # ensure order


    for key_a, key_b in zip(layer_keys_A, layer_keys_B):
        A = updated_A[key_a]
        B = updated_B[key_b]
        rotatedA[key_a], rotatedB[key_b] = rotation_align_random(A, B)

    return rotatedA, rotatedB


def rotation_alignment_soft(initial_model_ref: dict,
                       align: str,
                       updated_A: dict,
                       updated_B: dict,
                        rotation_lambda: float) -> (dict, dict):
    # soft rotation alignment
    rotatedA = OrderedDict()
    rotatedB = OrderedDict()
    layer_keys_A = list(updated_A.keys())
    # Create corresponding B keys (e.g., 'layer1.lora_A' -> 'layer1.lora_B')
    layer_keys_B = [key.replace('.lora_A', '.lora_B') for key in layer_keys_A]  # ensure order

    if align == 'A':
        ref_keys = layer_keys_A
    elif align == 'B':
        ref_keys = layer_keys_B
    else:
        raise ValueError("align must be 'A' or 'B'")


    for key_a, key_b, key_ref in zip(layer_keys_A, layer_keys_B, ref_keys):
        # Check if the keys exist before processing
        if key_a not in updated_A or key_b not in updated_B or key_ref not in initial_model_ref:
            print(f"Warning: Skipping layer due to missing keys. Looked for {key_a}, {key_b}, {key_ref}")
            continue

        A = updated_A[key_a]
        B = updated_B[key_b]
        model_ref = initial_model_ref[key_ref]


        rotatedA[key_a], rotatedB[key_b] = rotation_align_optimization_soft(model_ref, align, A, B, rotation_lambda)


    return rotatedA, rotatedB


def _closed_form_scaling(Q: torch.Tensor, Q_ref: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """c* = <Q, Q_ref> / ||Q||_F^2, computed in fp32 accumulation without fp32 copies."""
    if Q.shape != Q_ref.shape:
        raise ValueError(f"Shape mismatch: {Q.shape} vs {Q_ref.shape}")

    denom = torch.sum(Q * Q, dtype=torch.float32)
    denom = torch.clamp(denom, min=eps)
    numer = torch.sum(Q * Q_ref, dtype=torch.float32)
    return numer / denom  # float32 scalar on device


def scaling_align_optimization_inplace(
    initial_model_ref: torch.Tensor,
    align_matrix: str,
    updated_A: torch.Tensor,
    updated_B: torch.Tensor,
    eps: float = 1e-12,
) -> (torch.Tensor, torch.Tensor):
    """
    Hard scalar rescaling, alternating on A/B, preserving ΔW = B@A exactly.
    In-place update to avoid extra allocations.
    """

    if align_matrix not in ("A", "B"):
        raise ValueError("align_matrix must be 'A' or 'B'")

    with torch.no_grad():
        if align_matrix == "A":
            c = _closed_form_scaling(updated_A, initial_model_ref, eps=eps)
        else:
            c = _closed_form_scaling(updated_B, initial_model_ref, eps=eps)

        # safe inverse; keep sign to match unconstrained optimum
        abs_c = torch.clamp(torch.abs(c), min=eps)
        c_safe = torch.sign(c) * abs_c  # float32 scalar tensor

        # apply gauge-consistent scaling (in-place)
        if align_matrix == "A":
            updated_A.mul_(c_safe.to(dtype=updated_A.dtype))
            updated_B.div_(c_safe.to(dtype=updated_B.dtype))
        else:
            updated_B.mul_(c_safe.to(dtype=updated_B.dtype))
            updated_A.div_(c_safe.to(dtype=updated_A.dtype))

    return updated_A, updated_B


def scaling_alignment(
    initial_model_ref: dict,
    align: str,
    updated_A: dict,
    updated_B: dict,
    eps: float = 1e-12,
) -> (dict, dict):
    """
    Wrapper matching your rotation_alignment_soft structure.
    This version updates tensors in-place and returns the same dicts (or OrderedDict views).
    """
    scaledA = OrderedDict()
    scaledB = OrderedDict()

    layer_keys_A = list(updated_A.keys())
    layer_keys_B = [key.replace('.lora_A', '.lora_B') for key in layer_keys_A]

    if align == 'A':
        ref_keys = layer_keys_A
    elif align == 'B':
        ref_keys = layer_keys_B
    else:
        raise ValueError("align must be 'A' or 'B'")

    for key_a, key_b, key_ref in zip(layer_keys_A, layer_keys_B, ref_keys):
        if key_a not in updated_A or key_b not in updated_B or key_ref not in initial_model_ref:
            print(f"Warning: Skipping layer due to missing keys. Looked for {key_a}, {key_b}, {key_ref}")
            continue

        A = updated_A[key_a]
        B = updated_B[key_b]
        ref = initial_model_ref[key_ref]

        A2, B2 = scaling_align_optimization_inplace(ref, align, A, B, eps=eps)
        scaledA[key_a] = A2
        scaledB[key_b] = B2

    return scaledA, scaledB
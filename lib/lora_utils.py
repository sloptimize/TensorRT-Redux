"""
LoRA utility functions for loading, applying, and computing weight deltas.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save

logger = logging.getLogger(__name__)


def load_lora_weights(lora_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load LoRA weights from a safetensors or PyTorch file.

    Args:
        lora_path: Path to LoRA file (.safetensors or .pt)

    Returns:
        Dict of LoRA weights
    """
    logger.info(f"Loading LoRA weights from {lora_path}")

    if lora_path.suffix == ".safetensors":
        weights = {}
        with safe_open(str(lora_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    elif lora_path.suffix in [".pt", ".pth", ".bin"]:
        weights = torch.load(str(lora_path), map_location="cpu")
    else:
        raise ValueError(f"Unsupported LoRA file format: {lora_path.suffix}")

    logger.info(f"Loaded {len(weights)} LoRA tensors")
    return weights


def parse_lora_keys(lora_weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Parse LoRA keys into structured format.

    LoRA weights typically have names like:
    - lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
    - lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight

    Args:
        lora_weights: Raw LoRA weights dict

    Returns:
        Dict mapping base weight names to their LoRA components:
        {
            "base_weight_name": {
                "lora_down": tensor,
                "lora_up": tensor,
                "alpha": tensor (optional)
            }
        }
    """
    parsed = {}

    for key, tensor in lora_weights.items():
        # Extract base name and component type
        if ".lora_down" in key:
            base_name = key.replace(".lora_down.weight", "").replace(".lora_down", "")
            component = "lora_down"
        elif ".lora_up" in key:
            base_name = key.replace(".lora_up.weight", "").replace(".lora_up", "")
            component = "lora_up"
        elif ".alpha" in key:
            base_name = key.replace(".alpha", "")
            component = "alpha"
        else:
            # Skip non-LoRA weights
            continue

        if base_name not in parsed:
            parsed[base_name] = {}
        parsed[base_name][component] = tensor

    logger.info(f"Parsed {len(parsed)} LoRA weight pairs")
    return parsed


def compute_lora_merged_weight(
    original_weight: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute the merged weight with LoRA applied.

    LoRA formula: W' = W + scale * (alpha/rank) * (up @ down)

    Args:
        original_weight: Original model weight
        lora_down: LoRA down projection (rank x in_features)
        lora_up: LoRA up projection (out_features x rank)
        alpha: LoRA alpha scaling factor
        scale: Additional user-specified scale

    Returns:
        Merged weight tensor
    """
    # Get rank from down projection
    rank = lora_down.shape[0]

    # Compute alpha scaling
    if alpha is not None:
        alpha_value = alpha.item() if alpha.numel() == 1 else alpha[0].item()
    else:
        alpha_value = rank  # Default alpha = rank (no scaling)

    # LoRA delta: (alpha/rank) * scale * (up @ down)
    lora_delta = (alpha_value / rank) * scale * (lora_up @ lora_down)

    # Handle shape differences (conv vs linear)
    if len(original_weight.shape) == 4:
        # Conv2d: reshape delta to match
        lora_delta = lora_delta.view(original_weight.shape)

    return original_weight + lora_delta.to(original_weight.dtype)


def apply_lora_to_state_dict(
    state_dict: Dict[str, torch.Tensor],
    lora_weights: Dict[str, torch.Tensor],
    scale: float = 1.0,
    key_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply LoRA weights to a model state dict.

    Args:
        state_dict: Original model state dict
        lora_weights: LoRA weights (raw format from file)
        scale: LoRA strength/scale factor
        key_mapping: Optional mapping from LoRA keys to state dict keys

    Returns:
        Modified state dict with LoRA applied
    """
    logger.info(f"Applying LoRA with scale={scale}")

    # Parse LoRA weights into structured format
    parsed_lora = parse_lora_keys(lora_weights)

    # Create a copy of state dict
    merged_state_dict = {k: v.clone() for k, v in state_dict.items()}

    applied_count = 0

    for lora_name, lora_components in parsed_lora.items():
        if "lora_down" not in lora_components or "lora_up" not in lora_components:
            continue

        # Find matching state dict key
        target_key = find_matching_state_dict_key(lora_name, state_dict.keys(), key_mapping)

        if target_key is None:
            logger.debug(f"No match found for LoRA key: {lora_name}")
            continue

        # Get original weight
        original_weight = merged_state_dict[target_key]

        # Compute merged weight
        merged_weight = compute_lora_merged_weight(
            original_weight,
            lora_components["lora_down"],
            lora_components["lora_up"],
            lora_components.get("alpha"),
            scale,
        )

        merged_state_dict[target_key] = merged_weight
        applied_count += 1

    logger.info(f"Applied LoRA to {applied_count} weights")
    return merged_state_dict


def find_matching_state_dict_key(
    lora_name: str,
    state_dict_keys: List[str],
    key_mapping: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Find the state dict key that matches a LoRA weight name.

    LoRA names often differ from model names, e.g.:
    - LoRA: lora_unet_down_blocks_0_attentions_0_...to_q
    - Model: down_blocks.0.attentions.0...to_q.weight

    Args:
        lora_name: LoRA weight name (without .lora_down/.lora_up)
        state_dict_keys: List of state dict keys
        key_mapping: Optional explicit mapping

    Returns:
        Matching state dict key or None
    """
    if key_mapping and lora_name in key_mapping:
        return key_mapping[lora_name]

    # Convert LoRA name to potential model name
    # Remove common prefixes
    name = lora_name
    for prefix in ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Convert underscores to dots for nested modules
    # But be careful with numbered parts (down_blocks_0 -> down_blocks.0)
    parts = name.split("_")
    converted_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]
        # Check if next part is a number
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            converted_parts.append(f"{part}.{parts[i + 1]}")
            i += 2
        else:
            converted_parts.append(part)
            i += 1

    # Join with dots
    model_name = ".".join(converted_parts)

    # Try to find match
    for key in state_dict_keys:
        # Direct match
        if model_name in key:
            return key
        # Try with .weight suffix
        if f"{model_name}.weight" == key:
            return key

    return None


def compute_weight_deltas(
    original_weights: Dict[str, np.ndarray],
    modified_weights: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """
    Compute deltas between original and modified weights.

    Only returns weights that actually changed (by hash comparison).

    Args:
        original_weights: Original ONNX weights (numpy)
        modified_weights: Modified weights after LoRA (torch tensors)

    Returns:
        Dict of weight deltas (only changed weights)
    """
    from .onnx_utils import compute_weight_hash

    deltas = {}

    for name, original in original_weights.items():
        if name not in modified_weights:
            continue

        modified = modified_weights[name]
        if isinstance(modified, torch.Tensor):
            modified = modified.detach().cpu().numpy()

        # Ensure same dtype
        modified = modified.astype(original.dtype)

        # Check if changed
        original_hash = compute_weight_hash(original)
        modified_hash = compute_weight_hash(modified)

        if original_hash != modified_hash:
            # Store the full modified weight (not delta) for refitting
            deltas[name] = modified

    logger.info(f"Found {len(deltas)} changed weights")
    return deltas


def save_lora_deltas(
    deltas: Dict[str, np.ndarray],
    output_path: Path,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save LoRA weight deltas to a safetensors file.

    Args:
        deltas: Dict of weight deltas
        output_path: Path to save .trt_lora file
        metadata: Optional metadata to include
    """
    # Convert numpy arrays to torch tensors
    tensors = {name: torch.from_numpy(arr) for name, arr in deltas.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_save(tensors, str(output_path), metadata=metadata)

    logger.info(f"Saved LoRA deltas to {output_path}")


def load_lora_deltas(delta_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load LoRA weight deltas from a safetensors file.

    Args:
        delta_path: Path to .trt_lora file

    Returns:
        Dict of weight deltas
    """
    deltas = {}
    with safe_open(str(delta_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            deltas[key] = f.get_tensor(key)

    logger.info(f"Loaded {len(deltas)} weight deltas from {delta_path}")
    return deltas


def merge_multiple_lora_deltas(
    delta_files: List[Tuple[Path, float]],
) -> Dict[str, torch.Tensor]:
    """
    Merge multiple LoRA delta files with their scales.

    Args:
        delta_files: List of (path, scale) tuples

    Returns:
        Merged weight deltas
    """
    if not delta_files:
        return {}

    # Load first delta file as base
    merged = {}

    for delta_path, scale in delta_files:
        deltas = load_lora_deltas(delta_path)

        for name, delta in deltas.items():
            if name in merged:
                # Weighted addition for stacking
                merged[name] = merged[name] + scale * delta
            else:
                merged[name] = scale * delta

    logger.info(f"Merged {len(delta_files)} LoRA delta files")
    return merged

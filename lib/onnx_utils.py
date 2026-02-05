"""
ONNX utility functions for weight extraction and mapping.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import onnx
import torch

logger = logging.getLogger(__name__)


def extract_weight_mapping(onnx_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Extract weight names and metadata from an ONNX model.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Dict mapping weight names to their metadata:
        {
            "weight_name": {
                "shape": [dim1, dim2, ...],
                "dtype": "float16",
                "hash": "abc123..."
            }
        }
    """
    logger.info(f"Extracting weight mapping from {onnx_path}")

    model = onnx.load(str(onnx_path))
    mapping = {}

    for initializer in model.graph.initializer:
        name = initializer.name
        shape = list(initializer.dims)
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type)

        # Compute hash for change detection
        weight_data = numpy_from_tensor(initializer)
        weight_hash = compute_weight_hash(weight_data)

        mapping[name] = {
            "shape": shape,
            "dtype": dtype,
            "hash": weight_hash,
        }

    logger.info(f"Extracted {len(mapping)} weights from ONNX model")
    return mapping


def load_onnx_initializers(onnx_path: Path) -> Dict[str, np.ndarray]:
    """
    Load all initializers (weights) from an ONNX model.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Dict mapping weight names to numpy arrays
    """
    logger.info(f"Loading ONNX initializers from {onnx_path}")

    model = onnx.load(str(onnx_path))
    weights = {}

    for initializer in model.graph.initializer:
        name = initializer.name
        weights[name] = numpy_from_tensor(initializer)

    logger.info(f"Loaded {len(weights)} initializers")
    return weights


def numpy_from_tensor(tensor: onnx.TensorProto) -> np.ndarray:
    """Convert ONNX TensorProto to numpy array."""
    return onnx.numpy_helper.to_array(tensor)


def compute_weight_hash(weight: np.ndarray) -> str:
    """Compute a hash of weight data for change detection."""
    return hashlib.md5(weight.tobytes()).hexdigest()


def save_weight_mapping(mapping: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """Save weight mapping to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved weight mapping to {output_path}")


def load_weight_mapping(mapping_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load weight mapping from JSON file."""
    with open(mapping_path, "r") as f:
        return json.load(f)


def create_pytorch_to_onnx_mapping(
    pytorch_state_dict: Dict[str, torch.Tensor],
    onnx_weights: Dict[str, np.ndarray]
) -> Dict[str, str]:
    """
    Create a mapping from PyTorch weight names to ONNX weight names.

    This is done by matching shapes and values since ONNX export
    may rename weights.

    Args:
        pytorch_state_dict: PyTorch model state dict
        onnx_weights: Dict of ONNX weights

    Returns:
        Dict mapping PyTorch names to ONNX names
    """
    mapping = {}

    # Build lookup by shape and hash
    onnx_by_shape: Dict[Tuple[int, ...], List[Tuple[str, str]]] = {}
    for name, weight in onnx_weights.items():
        shape = tuple(weight.shape)
        weight_hash = compute_weight_hash(weight)
        if shape not in onnx_by_shape:
            onnx_by_shape[shape] = []
        onnx_by_shape[shape].append((name, weight_hash))

    for pt_name, pt_weight in pytorch_state_dict.items():
        pt_array = pt_weight.detach().cpu().numpy()
        pt_shape = tuple(pt_array.shape)
        pt_hash = compute_weight_hash(pt_array.astype(np.float16))

        if pt_shape in onnx_by_shape:
            for onnx_name, onnx_hash in onnx_by_shape[pt_shape]:
                # Match by hash
                if pt_hash == onnx_hash:
                    mapping[pt_name] = onnx_name
                    break
            else:
                # No exact hash match, try closest shape match
                # This can happen due to precision differences
                candidates = onnx_by_shape[pt_shape]
                if len(candidates) == 1:
                    mapping[pt_name] = candidates[0][0]

    logger.info(f"Mapped {len(mapping)}/{len(pytorch_state_dict)} PyTorch weights to ONNX")
    return mapping


def get_lora_target_weights(onnx_weights: Dict[str, np.ndarray]) -> List[str]:
    """
    Identify weights that are typically targeted by LoRA.

    LoRA typically modifies attention projection weights (q, k, v, out)
    and sometimes feedforward layers.

    Args:
        onnx_weights: Dict of ONNX weights

    Returns:
        List of weight names that LoRA would modify
    """
    lora_targets = []

    # Common patterns for attention weights
    attention_patterns = [
        "to_q", "to_k", "to_v", "to_out",
        "query", "key", "value",
        "proj_in", "proj_out",
        "ff.net",  # Feedforward in transformers
    ]

    for name in onnx_weights.keys():
        name_lower = name.lower()
        for pattern in attention_patterns:
            if pattern in name_lower:
                lora_targets.append(name)
                break

    logger.info(f"Identified {len(lora_targets)} potential LoRA target weights")
    return lora_targets


def export_model_to_onnx(
    model: torch.nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    output_path: Path,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        dummy_inputs: Dict of dummy input tensors
        output_path: Path to save ONNX file
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axis specifications
        opset_version: ONNX opset version
    """
    logger.info(f"Exporting model to ONNX: {output_path}")

    # Prepare inputs as tuple in correct order
    inputs = tuple(dummy_inputs[name] for name in input_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info(f"ONNX export complete: {output_path}")


def parse_onnx_to_trt_network(
    onnx_path: Path,
    builder: "trt.Builder",
    network: "trt.INetworkDefinition",
) -> bool:
    """
    Parse an ONNX model into a TensorRT network.

    Args:
        onnx_path: Path to ONNX file
        builder: TensorRT builder
        network: TensorRT network to populate

    Returns:
        True if parsing succeeded
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            return False

    return True

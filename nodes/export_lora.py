"""
TRT_LORA_EXPORT node for pre-computing LoRA weight deltas for TensorRT refitting.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
import numpy as np

# ComfyUI imports
import folder_paths
import comfy.model_management
import comfy.sd
import comfy.utils

from ..lib.onnx_utils import (
    load_onnx_initializers,
    load_weight_mapping,
)
from ..lib.lora_utils import (
    load_lora_weights,
    apply_lora_to_state_dict,
    compute_weight_deltas,
    save_lora_deltas,
)

logger = logging.getLogger(__name__)

# Register trt_lora folder
if "trt_lora" not in folder_paths.folder_names_and_paths:
    trt_lora_path = os.path.join(folder_paths.models_dir, "trt_lora")
    os.makedirs(trt_lora_path, exist_ok=True)
    folder_paths.folder_names_and_paths["trt_lora"] = ([trt_lora_path], {".trt_lora"})


def get_tensorrt_engines() -> List[str]:
    """Get list of available TensorRT engines."""
    engine_dir = os.path.join(folder_paths.models_dir, "tensorrt")
    if not os.path.exists(engine_dir):
        return []

    engines = []
    for root, dirs, files in os.walk(engine_dir):
        for f in files:
            if f.endswith(".engine"):
                rel_path = os.path.relpath(os.path.join(root, f), engine_dir)
                engines.append(rel_path)
    return sorted(engines)


def get_lora_files() -> List[str]:
    """Get list of available LoRA files."""
    lora_dir = folder_paths.get_folder_paths("loras")[0] if folder_paths.get_folder_paths("loras") else None
    if not lora_dir or not os.path.exists(lora_dir):
        return []

    loras = []
    for root, dirs, files in os.walk(lora_dir):
        for f in files:
            if f.endswith((".safetensors", ".pt", ".pth", ".bin")):
                rel_path = os.path.relpath(os.path.join(root, f), lora_dir)
                loras.append(rel_path)
    return sorted(loras)


class TRT_LORA_EXPORT:
    """
    Pre-compute LoRA weight deltas for TensorRT engine refitting.

    This node takes a base TensorRT engine (and its associated ONNX file)
    and a LoRA file, then computes the weight deltas needed for refitting.
    The deltas are saved as a .trt_lora file that can be loaded at runtime.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "base_engine": (get_tensorrt_engines(),),
                "lora_file": (get_lora_files(),),
                "output_name": ("STRING", {"default": "lora_deltas"}),
            },
            "optional": {
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("delta_path",)
    OUTPUT_NODE = True
    FUNCTION = "export_lora_deltas"
    CATEGORY = "TensorRT-Redux"

    def export_lora_deltas(
        self,
        base_engine: str,
        lora_file: str,
        output_name: str,
        lora_strength: float = 1.0,
    ) -> Tuple[str]:
        """Export LoRA weight deltas for TensorRT refitting."""

        # Resolve paths
        engine_dir = os.path.join(folder_paths.models_dir, "tensorrt")
        engine_path = Path(engine_dir) / base_engine

        # Find associated ONNX and weight mapping files
        base_name = engine_path.stem
        onnx_path = engine_path.with_suffix(".onnx")
        mapping_path = engine_path.parent / f"{base_name}_weights.json"

        if not onnx_path.exists():
            # Try without shape suffix
            onnx_candidates = list(engine_path.parent.glob(f"*.onnx"))
            if onnx_candidates:
                onnx_path = onnx_candidates[0]
            else:
                raise FileNotFoundError(
                    f"ONNX file not found for engine {base_engine}. "
                    "Please ensure the engine was exported with TRT_MODEL_EXPORT."
                )

        if not mapping_path.exists():
            mapping_candidates = list(engine_path.parent.glob(f"*_weights.json"))
            if mapping_candidates:
                mapping_path = mapping_candidates[0]

        # Load LoRA file
        lora_dir = folder_paths.get_folder_paths("loras")[0]
        lora_path = Path(lora_dir) / lora_file
        logger.info(f"Loading LoRA from {lora_path}")
        lora_weights = load_lora_weights(lora_path)

        # Load original ONNX weights
        logger.info(f"Loading ONNX weights from {onnx_path}")
        original_weights = load_onnx_initializers(onnx_path)

        # We need to apply LoRA to get the modified weights
        # This requires loading the base model temporarily to get the state dict
        # For now, we'll compute deltas directly from ONNX + LoRA

        # Create a pseudo state dict from ONNX weights for LoRA application
        state_dict = {
            name: torch.from_numpy(weight) for name, weight in original_weights.items()
        }

        # Apply LoRA to get modified state dict
        logger.info(f"Applying LoRA with strength={lora_strength}")
        modified_state_dict = apply_lora_to_state_dict(
            state_dict,
            lora_weights,
            scale=lora_strength,
        )

        # Compute deltas (only changed weights)
        logger.info("Computing weight deltas...")
        deltas = compute_weight_deltas(original_weights, modified_state_dict)

        if not deltas:
            logger.warning("No weight changes detected. LoRA may not match this model architecture.")
            # Create empty delta file anyway
            deltas = {}

        # Save deltas
        output_dir = os.path.join(folder_paths.models_dir, "trt_lora")
        os.makedirs(output_dir, exist_ok=True)

        # Include LoRA name and strength in filename
        lora_name = Path(lora_file).stem
        output_filename = f"{output_name}_{lora_name}_s{lora_strength:.2f}.trt_lora"
        output_path = Path(output_dir) / output_filename

        metadata = {
            "base_engine": base_engine,
            "lora_file": lora_file,
            "lora_strength": str(lora_strength),
            "num_deltas": str(len(deltas)),
        }

        save_lora_deltas(deltas, output_path, metadata)

        logger.info(f"Saved {len(deltas)} weight deltas to {output_path}")

        return (str(output_path),)


class TRT_LORA_EXPORT_FROM_MODEL:
    """
    Export LoRA deltas using a loaded ComfyUI model for more accurate weight matching.

    This version loads the actual model to ensure correct weight name mapping
    between PyTorch and ONNX formats.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "base_engine": (get_tensorrt_engines(),),
                "lora_file": (get_lora_files(),),
                "output_name": ("STRING", {"default": "lora_deltas"}),
            },
            "optional": {
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("delta_path",)
    OUTPUT_NODE = True
    FUNCTION = "export_lora_deltas"
    CATEGORY = "TensorRT-Redux"

    def export_lora_deltas(
        self,
        model,
        base_engine: str,
        lora_file: str,
        output_name: str,
        lora_strength: float = 1.0,
    ) -> Tuple[str]:
        """Export LoRA weight deltas using the actual model state dict."""

        # Resolve paths
        engine_dir = os.path.join(folder_paths.models_dir, "tensorrt")
        engine_path = Path(engine_dir) / base_engine
        onnx_path = engine_path.with_suffix(".onnx")

        if not onnx_path.exists():
            onnx_candidates = list(engine_path.parent.glob(f"*.onnx"))
            if onnx_candidates:
                onnx_path = onnx_candidates[0]
            else:
                raise FileNotFoundError(f"ONNX file not found for engine {base_engine}")

        # Load LoRA
        lora_dir = folder_paths.get_folder_paths("loras")[0]
        lora_path = Path(lora_dir) / lora_file
        lora_weights = load_lora_weights(lora_path)

        # Get original model state dict
        unet = model.model.diffusion_model
        original_state_dict = {k: v.clone() for k, v in unet.state_dict().items()}

        # Apply LoRA to model
        logger.info(f"Applying LoRA with strength={lora_strength}")
        modified_state_dict = apply_lora_to_state_dict(
            original_state_dict,
            lora_weights,
            scale=lora_strength,
        )

        # Load ONNX weights for comparison
        original_onnx = load_onnx_initializers(onnx_path)

        # Create mapping from PyTorch to ONNX names
        # This is done by matching shapes and values
        from ..lib.onnx_utils import create_pytorch_to_onnx_mapping
        pt_to_onnx = create_pytorch_to_onnx_mapping(original_state_dict, original_onnx)

        # Convert modified weights to ONNX names
        modified_onnx = {}
        for pt_name, tensor in modified_state_dict.items():
            if pt_name in pt_to_onnx:
                onnx_name = pt_to_onnx[pt_name]
                modified_onnx[onnx_name] = tensor

        # Compute deltas
        deltas = compute_weight_deltas(original_onnx, modified_onnx)

        # Save deltas
        output_dir = os.path.join(folder_paths.models_dir, "trt_lora")
        os.makedirs(output_dir, exist_ok=True)

        lora_name = Path(lora_file).stem
        output_filename = f"{output_name}_{lora_name}_s{lora_strength:.2f}.trt_lora"
        output_path = Path(output_dir) / output_filename

        metadata = {
            "base_engine": base_engine,
            "lora_file": lora_file,
            "lora_strength": str(lora_strength),
            "num_deltas": str(len(deltas)),
        }

        save_lora_deltas(deltas, output_path, metadata)

        logger.info(f"Saved {len(deltas)} weight deltas to {output_path}")

        return (str(output_path),)

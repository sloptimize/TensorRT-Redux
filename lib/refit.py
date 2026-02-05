"""
TensorRT engine refitting logic for LoRA support.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import tensorrt as trt

from .trt_utils import check_refit_support, trt_dtype_to_torch

logger = logging.getLogger(__name__)


class EngineRefitter:
    """
    Handles TensorRT engine refitting for LoRA weight updates.

    This class manages:
    - Loading and caching weight deltas
    - Hash-based change detection to skip unnecessary refits
    - Applying weight updates via TensorRT refitting API
    """

    def __init__(self, engine: trt.ICudaEngine):
        """
        Initialize the refitter.

        Args:
            engine: TensorRT engine (must have been built with REFIT flag)
        """
        if not check_refit_support(engine):
            raise ValueError(
                "Engine was not built with REFIT flag. "
                "Please re-export the model with enable_refit=True"
            )

        self.engine = engine
        self.refitter = trt.Refitter(engine, trt.Logger(trt.Logger.WARNING))
        self._current_lora_hash: Optional[str] = None
        self._refitted_weights: Set[str] = set()
        self._original_weights: Dict[str, np.ndarray] = {}

    def get_refittable_weights(self) -> List[str]:
        """Get list of weight names that can be refitted."""
        return list(self.refitter.get_all_weights())

    def cache_original_weights(self, onnx_weights: Dict[str, np.ndarray]) -> None:
        """
        Cache original weights for potential reset.

        Args:
            onnx_weights: Original ONNX weights
        """
        refittable = set(self.get_refittable_weights())

        for name, weight in onnx_weights.items():
            if name in refittable:
                self._original_weights[name] = weight.copy()

        logger.info(f"Cached {len(self._original_weights)} original weights")

    def compute_lora_hash(self, lora_config: List[Tuple[Path, float]]) -> str:
        """
        Compute a hash of the current LoRA configuration.

        Args:
            lora_config: List of (lora_path, scale) tuples

        Returns:
            Hash string for the configuration
        """
        config_str = "|".join(
            f"{path}:{scale}" for path, scale in sorted(lora_config, key=lambda x: str(x[0]))
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def needs_refit(self, lora_config: List[Tuple[Path, float]]) -> bool:
        """
        Check if engine needs refitting for the given LoRA config.

        Args:
            lora_config: List of (lora_path, scale) tuples

        Returns:
            True if refit is needed
        """
        if not lora_config:
            # No LoRA - check if we need to reset
            return self._current_lora_hash is not None

        new_hash = self.compute_lora_hash(lora_config)
        return new_hash != self._current_lora_hash

    def refit_with_weights(
        self,
        weight_updates: Dict[str, np.ndarray],
        dtype: torch.dtype = torch.float16
    ) -> bool:
        """
        Apply weight updates to the engine.

        Args:
            weight_updates: Dict mapping weight names to new values
            dtype: Target dtype for weights

        Returns:
            True if refit succeeded
        """
        if not weight_updates:
            return True

        logger.info(f"Refitting engine with {len(weight_updates)} weight updates")

        refittable = set(self.get_refittable_weights())
        applied = 0

        for name, weight in weight_updates.items():
            if name not in refittable:
                logger.debug(f"Weight {name} is not refittable, skipping")
                continue

            # Convert to correct dtype
            if dtype == torch.float16:
                weight = weight.astype(np.float16)
            elif dtype == torch.bfloat16:
                # numpy doesn't have bfloat16, use float32 as intermediate
                weight = weight.astype(np.float32)

            # Set the weight
            success = self.refitter.set_named_weights(name, weight)
            if not success:
                logger.warning(f"Failed to set weight: {name}")
            else:
                self._refitted_weights.add(name)
                applied += 1

        # Check for missing weights
        missing = self.refitter.get_missing_weights()
        if missing:
            logger.warning(f"Missing weights after refit: {missing}")

        # Apply the refit
        success = self.refitter.refit_cuda_engine()

        if success:
            logger.info(f"Successfully refitted {applied} weights")
        else:
            logger.error("Engine refit failed")

        return success

    def apply_lora(
        self,
        lora_config: List[Tuple[Path, float]],
        dtype: torch.dtype = torch.float16
    ) -> bool:
        """
        Apply LoRA configuration to the engine.

        Args:
            lora_config: List of (lora_delta_path, scale) tuples
            dtype: Target dtype

        Returns:
            True if successful
        """
        from .lora_utils import merge_multiple_lora_deltas

        if not self.needs_refit(lora_config):
            logger.info("LoRA config unchanged, skipping refit")
            return True

        if not lora_config:
            # Reset to original weights
            return self.reset_to_original()

        # Merge LoRA deltas
        merged_deltas = merge_multiple_lora_deltas(lora_config)

        # Convert torch tensors to numpy
        weight_updates = {
            name: tensor.numpy() for name, tensor in merged_deltas.items()
        }

        # Apply refit
        success = self.refit_with_weights(weight_updates, dtype)

        if success:
            self._current_lora_hash = self.compute_lora_hash(lora_config)

        return success

    def reset_to_original(self) -> bool:
        """
        Reset engine to original weights (remove LoRA).

        Returns:
            True if successful
        """
        if not self._original_weights:
            logger.warning("No original weights cached, cannot reset")
            return False

        logger.info("Resetting engine to original weights")

        # Only reset weights that were modified
        weights_to_reset = {
            name: weight
            for name, weight in self._original_weights.items()
            if name in self._refitted_weights
        }

        success = self.refit_with_weights(weights_to_reset)

        if success:
            self._current_lora_hash = None
            self._refitted_weights.clear()

        return success


def refit_engine_with_lora(
    engine: trt.ICudaEngine,
    lora_deltas: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16
) -> bool:
    """
    Simple function to refit an engine with LoRA deltas.

    Args:
        engine: TensorRT engine (must be refittable)
        lora_deltas: Dict of weight updates
        dtype: Target dtype

    Returns:
        True if successful
    """
    refitter = EngineRefitter(engine)

    weight_updates = {
        name: tensor.numpy() for name, tensor in lora_deltas.items()
    }

    return refitter.refit_with_weights(weight_updates, dtype)


def get_engine_weights(engine: trt.ICudaEngine) -> Dict[str, np.ndarray]:
    """
    Extract current weights from a TensorRT engine.

    Args:
        engine: TensorRT engine

    Returns:
        Dict of weight names to numpy arrays
    """
    if not check_refit_support(engine):
        raise ValueError("Engine must be refittable to extract weights")

    refitter = trt.Refitter(engine, trt.Logger(trt.Logger.WARNING))
    weights = {}

    for name in refitter.get_all_weights():
        # Get weight data through the refitter
        # Note: This requires TensorRT 10.0+ for get_named_weights
        try:
            weight = refitter.get_named_weights(name)
            if weight is not None:
                weights[name] = np.array(weight)
        except Exception as e:
            logger.debug(f"Could not get weight {name}: {e}")

    return weights

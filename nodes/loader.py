"""
TRT_LOADER_LORA node for loading TensorRT engines with optional LoRA refitting.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
import numpy as np
import tensorrt as trt

# ComfyUI imports
import folder_paths
import comfy.model_management
import comfy.model_base
import comfy.model_patcher
import comfy.supported_models

from ..lib.trt_utils import (
    load_engine,
    check_refit_support,
    get_engine_io_info,
    trt_dtype_to_torch,
)
from ..lib.refit import EngineRefitter
from ..lib.lora_utils import load_lora_deltas

logger = logging.getLogger(__name__)


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


def get_trt_lora_files() -> List[str]:
    """Get list of available TRT LoRA delta files."""
    lora_dir = os.path.join(folder_paths.models_dir, "trt_lora")
    if not os.path.exists(lora_dir):
        return ["none"]

    loras = ["none"]
    for root, dirs, files in os.walk(lora_dir):
        for f in files:
            if f.endswith(".trt_lora"):
                rel_path = os.path.relpath(os.path.join(root, f), lora_dir)
                loras.append(rel_path)
    return sorted(loras)


class TrTUnet(torch.nn.Module):
    """
    TensorRT UNet wrapper that mimics the ComfyUI UNet interface.

    Handles:
    - Loading and executing TensorRT engine
    - Dynamic batch splitting for larger batches
    - LoRA refitting via EngineRefitter
    """

    def __init__(
        self,
        engine_path: Path,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.engine_path = engine_path
        self.dtype = dtype

        # Load engine
        self.engine, self.context = load_engine(engine_path)

        # Check refit support
        self.refittable = check_refit_support(self.engine)
        if self.refittable:
            self.refitter = EngineRefitter(self.engine)
        else:
            self.refitter = None
            logger.warning("Engine was not built with REFIT flag - LoRA support disabled")

        # Get I/O info
        self.io_info = get_engine_io_info(self.engine)

        # Determine batch constraints from profile
        self._batch_min, self._batch_opt, self._batch_max = self._get_batch_constraints()

        logger.info(f"Loaded TRT engine: batch_range=[{self._batch_min}, {self._batch_max}]")

    def _get_batch_constraints(self) -> Tuple[int, int, int]:
        """Get batch size constraints from optimization profile."""
        try:
            # Get shape for first input (usually 'x')
            input_name = self.engine.get_tensor_name(0)
            shapes = self.engine.get_tensor_profile_shape(input_name, 0)
            return shapes[0][0], shapes[1][0], shapes[2][0]
        except Exception as e:
            logger.warning(f"Could not get batch constraints: {e}")
            return 1, 1, 4

    def apply_lora(self, lora_deltas: Dict[str, torch.Tensor], scale: float = 1.0) -> bool:
        """
        Apply LoRA deltas to the engine via refitting.

        Args:
            lora_deltas: Dict of weight deltas
            scale: Scale factor for deltas

        Returns:
            True if successful
        """
        if not self.refittable or self.refitter is None:
            logger.error("Engine does not support refitting")
            return False

        # Scale deltas
        if scale != 1.0:
            lora_deltas = {k: v * scale for k, v in lora_deltas.items()}

        # Convert to numpy for refitting
        weight_updates = {
            name: tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in lora_deltas.items()
        }

        return self.refitter.refit_with_weights(weight_updates, self.dtype)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through TensorRT engine.

        Handles batch splitting if batch_size > max_batch.
        """
        batch_size = x.shape[0]

        # Determine split factor
        split_factor = 1
        if batch_size > self._batch_max:
            for i in range(self._batch_max, self._batch_min - 1, -1):
                if batch_size % i == 0:
                    split_factor = batch_size // i
                    break

        # Prepare inputs
        model_inputs = {
            "x": x,
            "timesteps": timesteps,
            "context": context,
        }

        if y is not None:
            model_inputs["y"] = y

        # Handle extra inputs (e.g., Flux guidance)
        for key, value in kwargs.items():
            if key in [n for n in self.io_info["inputs"]]:
                model_inputs[key] = value

        # Set binding shapes
        for name, tensor in model_inputs.items():
            if name in self.io_info["inputs"]:
                shape = list(tensor.shape)
                if split_factor > 1:
                    shape[0] = shape[0] // split_factor
                self.context.set_input_shape(name, shape)

        # Prepare output buffer
        output_name = list(self.io_info["outputs"].keys())[0]
        output_shape = list(self.context.get_tensor_shape(output_name))

        # Handle dynamic output dimensions
        for idx, dim in enumerate(output_shape):
            if dim == -1:
                output_shape[idx] = x.shape[idx] if idx < len(x.shape) else 1

        if split_factor > 1:
            output_shape[0] *= split_factor

        output = torch.empty(
            output_shape,
            device=x.device,
            dtype=self.dtype
        )

        # Get CUDA stream
        stream = torch.cuda.current_stream(x.device)

        # Execute with potential batch splitting
        sub_batch = batch_size // split_factor

        for i in range(split_factor):
            start_idx = i * sub_batch
            end_idx = (i + 1) * sub_batch

            # Set input addresses
            for name, tensor in model_inputs.items():
                if name in self.io_info["inputs"]:
                    sub_tensor = tensor[start_idx:end_idx]
                    # Convert dtype if needed
                    input_dtype_str = self.io_info["inputs"][name]["dtype"]
                    if "HALF" in input_dtype_str:
                        sub_tensor = sub_tensor.half()
                    elif "BF16" in input_dtype_str:
                        sub_tensor = sub_tensor.bfloat16()

                    self.context.set_tensor_address(name, sub_tensor.data_ptr())

            # Set output address
            out_slice = output[start_idx:end_idx]
            self.context.set_tensor_address(output_name, out_slice.data_ptr())

            # Execute
            self.context.execute_async_v3(stream.cuda_stream)

        # Sync is handled by PyTorch stream management
        return output


class TRT_LOADER_LORA:
    """
    Load a TensorRT engine with optional LoRA refitting support.

    This node loads a pre-built TensorRT engine and optionally applies
    LoRA weight deltas via TensorRT's refitting API.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "engine_name": (get_tensorrt_engines(),),
                "model_type": ([
                    "sdxl_base",
                    "sdxl_refiner",
                    "sd1.x",
                    "sd2.x",
                    "sd3",
                    "flux_dev",
                    "flux_schnell",
                ],),
            },
            "optional": {
                "lora_delta": (get_trt_lora_files(),),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_engine"
    CATEGORY = "TensorRT-Redux"

    def load_engine(
        self,
        engine_name: str,
        model_type: str,
        lora_delta: str = "none",
        lora_scale: float = 1.0,
    ) -> Tuple[Any]:
        """Load TensorRT engine and wrap as ComfyUI MODEL."""

        # Resolve engine path
        engine_dir = os.path.join(folder_paths.models_dir, "tensorrt")
        engine_path = Path(engine_dir) / engine_name

        # Determine dtype from model type
        dtype = torch.bfloat16 if "flux" in model_type else torch.float16

        # Create TRT UNet wrapper
        trt_unet = TrTUnet(engine_path, dtype)

        # Apply LoRA if specified
        if lora_delta and lora_delta != "none":
            lora_dir = os.path.join(folder_paths.models_dir, "trt_lora")
            lora_path = Path(lora_dir) / lora_delta

            logger.info(f"Loading LoRA deltas from {lora_path}")
            deltas = load_lora_deltas(lora_path)

            if not trt_unet.apply_lora(deltas, lora_scale):
                logger.warning("Failed to apply LoRA to engine")

        # Create ComfyUI model wrapper
        model = self._create_model_wrapper(trt_unet, model_type)

        return (model,)

    def _create_model_wrapper(self, trt_unet: TrTUnet, model_type: str) -> Any:
        """Create a ComfyUI ModelPatcher with TRT UNet."""

        # Get appropriate model configuration
        if model_type == "sdxl_base":
            model_conf = comfy.supported_models.SDXL({})
            model_conf.unet_config["disable_unet_model_creation"] = True
            base_model = comfy.model_base.SDXL(model_conf)
        elif model_type == "sdxl_refiner":
            model_conf = comfy.supported_models.SDXLRefiner({})
            model_conf.unet_config["disable_unet_model_creation"] = True
            base_model = comfy.model_base.SDXLRefiner(model_conf)
        elif model_type == "sd1.x":
            model_conf = comfy.supported_models.SD15({})
            model_conf.unet_config["disable_unet_model_creation"] = True
            base_model = comfy.model_base.BaseModel(model_conf)
        elif model_type == "sd2.x":
            model_conf = comfy.supported_models.SD20({})
            model_conf.unet_config["disable_unet_model_creation"] = True
            base_model = comfy.model_base.BaseModel(model_conf)
        elif model_type == "sd3":
            model_conf = comfy.supported_models.SD3({})
            model_conf.unet_config["disable_unet_model_creation"] = True
            base_model = comfy.model_base.SD3(model_conf)
        elif model_type in ["flux_dev", "flux_schnell"]:
            model_conf = comfy.supported_models.Flux({})
            model_conf.unet_config["disable_unet_model_creation"] = True
            base_model = comfy.model_base.Flux(model_conf)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Replace diffusion model with TRT UNet
        base_model.diffusion_model = trt_unet

        # Create ModelPatcher
        model_patcher = comfy.model_patcher.ModelPatcher(
            base_model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=torch.device("cpu"),
        )

        return model_patcher


class TRT_LORA_STACK:
    """
    Stack multiple LoRA deltas for combined application.

    This node allows chaining multiple LoRA files with different scales.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "lora_delta": (get_trt_lora_files(),),
                "scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "lora_stack": ("TRT_LORA_STACK",),
            }
        }

    RETURN_TYPES = ("TRT_LORA_STACK",)
    FUNCTION = "stack_lora"
    CATEGORY = "TensorRT-Redux"

    def stack_lora(
        self,
        lora_delta: str,
        scale: float,
        lora_stack: Optional[List[Tuple[str, float]]] = None,
    ) -> Tuple[List[Tuple[str, float]]]:
        """Add LoRA to stack."""

        stack = list(lora_stack) if lora_stack else []

        if lora_delta and lora_delta != "none":
            stack.append((lora_delta, scale))

        return (stack,)


class TRT_LOADER_LORA_STACKED:
    """
    Load TensorRT engine with multiple stacked LoRAs.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "engine_name": (get_tensorrt_engines(),),
                "model_type": ([
                    "sdxl_base",
                    "sdxl_refiner",
                    "sd1.x",
                    "sd2.x",
                    "sd3",
                    "flux_dev",
                    "flux_schnell",
                ],),
                "lora_stack": ("TRT_LORA_STACK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_engine"
    CATEGORY = "TensorRT-Redux"

    def load_engine(
        self,
        engine_name: str,
        model_type: str,
        lora_stack: List[Tuple[str, float]],
    ) -> Tuple[Any]:
        """Load TensorRT engine with stacked LoRAs."""

        from ..lib.lora_utils import merge_multiple_lora_deltas

        # Resolve engine path
        engine_dir = os.path.join(folder_paths.models_dir, "tensorrt")
        engine_path = Path(engine_dir) / engine_name

        # Determine dtype
        dtype = torch.bfloat16 if "flux" in model_type else torch.float16

        # Create TRT UNet
        trt_unet = TrTUnet(engine_path, dtype)

        # Merge and apply LoRA stack
        if lora_stack:
            lora_dir = os.path.join(folder_paths.models_dir, "trt_lora")

            # Convert to full paths
            lora_paths = [
                (Path(lora_dir) / name, scale)
                for name, scale in lora_stack
            ]

            logger.info(f"Merging {len(lora_paths)} LoRA deltas")
            merged_deltas = merge_multiple_lora_deltas(lora_paths)

            if not trt_unet.apply_lora(merged_deltas, scale=1.0):
                logger.warning("Failed to apply merged LoRAs")

        # Create model wrapper
        loader = TRT_LOADER_LORA()
        model = loader._create_model_wrapper(trt_unet, model_type)

        return (model,)

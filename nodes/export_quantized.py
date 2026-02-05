"""
TRT_MODEL_EXPORT_QUANTIZED node for exporting quantized models to TensorRT.

Supports FP8 (Ada+) and NVFP4 (Blackwell) quantization using NVIDIA Model Optimizer.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import tensorrt as trt

# ComfyUI imports
import folder_paths
import comfy.model_management
import comfy.model_base
import comfy.supported_models

from ..lib.trt_utils import (
    create_builder_config,
    serialize_engine,
    save_timing_cache,
)
from ..lib.onnx_utils import (
    extract_weight_mapping,
    save_weight_mapping,
    parse_onnx_to_trt_network,
)
from ..lib.quantize import (
    QuantFormat,
    QuantConfig,
    check_quantization_available,
    quantize_unet,
    get_calibration_data,
    export_quantized_onnx,
)

logger = logging.getLogger(__name__)

# Register tensorrt folder with ComfyUI
if "tensorrt" not in folder_paths.folder_names_and_paths:
    tensorrt_path = os.path.join(folder_paths.models_dir, "tensorrt")
    os.makedirs(tensorrt_path, exist_ok=True)
    folder_paths.folder_names_and_paths["tensorrt"] = ([tensorrt_path], {".engine"})


def _check_modelopt_available() -> bool:
    """Check if nvidia-modelopt is installed."""
    try:
        import modelopt  # noqa: F401
        return True
    except ImportError:
        return False


class TRT_MODEL_EXPORT_QUANTIZED:
    """
    Export a ComfyUI model to TensorRT with FP8 or NVFP4 quantization.

    This node quantizes the model using NVIDIA Model Optimizer before
    exporting to ONNX and building the TensorRT engine.

    Requirements:
    - FP8: RTX 40 series+ (Ada Lovelace, SM >= 8.9)
    - NVFP4: RTX 50 series+ (Blackwell, SM >= 10.0) + CUDA 13.0
    - nvidia-modelopt package
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Check what quantization options are available
        quant_options = ["fp16"]  # Always available

        if _check_modelopt_available():
            # Check GPU capability
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                sm_version = capability[0] * 10 + capability[1]

                if sm_version >= 89:  # Ada Lovelace
                    quant_options.insert(0, "fp8")
                if sm_version >= 100:  # Blackwell
                    quant_options.insert(0, "nvfp4")

        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "tensorrt/model_quant"}),
                "quantization": (quant_options, {"default": quant_options[0]}),
                "batch_min": ("INT", {"default": 1, "min": 1, "max": 16}),
                "batch_opt": ("INT", {"default": 1, "min": 1, "max": 16}),
                "batch_max": ("INT", {"default": 4, "min": 1, "max": 16}),
                "height_min": ("INT", {"default": 832, "min": 256, "max": 4096, "step": 64}),
                "height_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height_max": ("INT", {"default": 1920, "min": 256, "max": 4096, "step": 64}),
                "width_min": ("INT", {"default": 832, "min": 256, "max": 4096, "step": 64}),
                "width_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_max": ("INT", {"default": 1920, "min": 256, "max": 4096, "step": 64}),
                "context_min": ("INT", {"default": 1, "min": 1, "max": 512}),
                "context_opt": ("INT", {"default": 77, "min": 1, "max": 512}),
                "context_max": ("INT", {"default": 154, "min": 1, "max": 512}),
            },
            "optional": {
                "enable_refit": ("BOOLEAN", {"default": True}),
                "calibration_steps": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "export_model"
    CATEGORY = "TensorRT-Redux"

    def export_model(
        self,
        model,
        filename_prefix: str,
        quantization: str,
        batch_min: int,
        batch_opt: int,
        batch_max: int,
        height_min: int,
        height_opt: int,
        height_max: int,
        width_min: int,
        width_opt: int,
        width_max: int,
        context_min: int,
        context_opt: int,
        context_max: int,
        enable_refit: bool = True,
        calibration_steps: int = 512,
    ) -> Tuple:
        """Export quantized model to TensorRT engine."""

        # Check quantization availability
        available = check_quantization_available()
        quant_format = QuantFormat(quantization)

        if quant_format in (QuantFormat.NVFP4, QuantFormat.FP8) and not available["modelopt"]:
            raise RuntimeError(
                f"{quantization.upper()} quantization requires nvidia-modelopt.\n\n"
                "Install with:\n"
                "  pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com\n\n"
                "After installing, restart ComfyUI."
            )

        if quant_format == QuantFormat.NVFP4 and not available["nvfp4"]:
            raise RuntimeError(
                "NVFP4 quantization requires Blackwell GPU (RTX 50 series, SM >= 10.0).\n"
                f"Your GPU: SM {torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}"
            )

        if quant_format == QuantFormat.FP8 and not available["fp8"]:
            raise RuntimeError(
                "FP8 quantization requires Ada Lovelace+ GPU (RTX 40 series, SM >= 8.9).\n"
                f"Your GPU: SM {torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}"
            )

        # Determine output paths
        output_dir = folder_paths.get_output_directory()
        prefix_parts = filename_prefix.rsplit("/", 1)
        if len(prefix_parts) > 1:
            subdir, base_name = prefix_parts
            output_dir = os.path.join(output_dir, subdir)
        else:
            base_name = filename_prefix

        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with quant info
        shape_suffix = f"_{quantization}_dyn-b{batch_min}-{batch_max}-h{height_min}-{height_max}-w{width_min}-{width_max}"
        engine_path = Path(output_dir) / f"{base_name}{shape_suffix}.engine"
        onnx_path = Path(output_dir) / f"{base_name}{shape_suffix}.onnx"
        mapping_path = Path(output_dir) / f"{base_name}{shape_suffix}_weights.json"

        # Determine dtype based on quantization
        if quant_format == QuantFormat.NVFP4:
            dtype = torch.bfloat16  # NVFP4 works best with BF16 base
            opset_version = 23  # Required for FP4 support
        elif quant_format == QuantFormat.FP8:
            dtype = torch.float16
            opset_version = 19  # FP8 support
        else:
            dtype = torch.float16
            opset_version = 17

        # Unload other models and force load target model
        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu(
            [model],
            force_patch_weights=True,
            force_full_load=True
        )

        # Extract UNet and model config
        unet = model.model.diffusion_model
        model_info = self._detect_model_type(model)
        logger.info(f"Detected model type: {model_info['type']}")
        logger.info(f"Quantization format: {quantization}")

        device = comfy.model_management.get_torch_device()

        # Convert model to target dtype to avoid mismatches
        logger.info(f"Converting model to {dtype}...")
        unet = unet.to(dtype=dtype)

        # Generate calibration data for PTQ
        if quant_format != QuantFormat.FP16:
            logger.info(f"Generating calibration data ({calibration_steps} samples)...")
            calibration_data = get_calibration_data(
                model_type=model_info["type"],
                batch_size=batch_opt,
                height=height_opt,
                width=width_opt,
                context_dim=model_info["context_dim"],
                context_len=context_opt,
                device=device,
                dtype=dtype,
                num_samples=calibration_steps,
            )

            # Quantize the UNet
            logger.info(f"Quantizing UNet to {quantization}...")
            quant_config = QuantConfig(
                format=quant_format,
                calibration_size=calibration_steps,
            )
            unet = quantize_unet(unet, quant_format, calibration_data, quant_config)
        else:
            calibration_data = None

        # Create dummy inputs for ONNX export
        dummy_inputs, input_names, output_names, dynamic_axes = self._create_dummy_inputs(
            model_info,
            batch_opt, height_opt, width_opt, context_opt,
            device, dtype
        )

        # Export to ONNX
        logger.info("Exporting quantized model to ONNX...")
        unet.eval()
        with torch.no_grad():
            export_quantized_onnx(
                unet,
                dummy_inputs,
                onnx_path,
                input_names,
                output_names,
                dynamic_axes,
                opset_version=opset_version,
            )

        # Extract and save weight mapping
        logger.info("Extracting weight mapping...")
        weight_mapping = extract_weight_mapping(onnx_path)
        save_weight_mapping(weight_mapping, mapping_path)

        # Build TensorRT engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        self._build_trt_engine(
            onnx_path,
            engine_path,
            input_names,
            model_info,
            batch_min, batch_opt, batch_max,
            height_min, height_opt, height_max,
            width_min, width_opt, width_max,
            context_min, context_opt, context_max,
            quant_format,
            dtype,
            enable_refit,
        )

        # Cleanup
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        logger.info(f"Quantized export complete!")
        logger.info(f"  Engine: {engine_path}")
        logger.info(f"  ONNX: {onnx_path}")
        logger.info(f"  Weights: {mapping_path}")
        logger.info(f"  Quantization: {quantization}")

        return ()

    def _detect_model_type(self, model) -> Dict[str, Any]:
        """Detect the model architecture type and extract relevant dimensions."""
        model_config = model.model.model_config
        unet_config = model_config.unet_config

        info = {
            "type": "unknown",
            "context_dim": unet_config.get("context_dim", 2048),
            "y_dim": model.model.adm_channels,
            "in_channels": unet_config.get("in_channels", 4),
            "uses_temporal": unet_config.get("use_temporal_resblock", False),
        }

        # Detect specific model types
        if isinstance(model.model, comfy.model_base.SDXL):
            info["type"] = "sdxl"
            info["context_dim"] = 2048
            info["y_dim"] = 2816
        elif isinstance(model.model, comfy.model_base.SD3):
            info["type"] = "sd3"
            context_embedder = unet_config.get("context_embedder_config", {})
            info["context_dim"] = context_embedder.get("params", {}).get("in_features", 4096)
        elif isinstance(model.model, comfy.model_base.Flux):
            info["type"] = "flux"
            info["context_dim"] = unet_config.get("context_in_dim", 4096)
            info["y_dim"] = unet_config.get("vec_in_dim", 768)
        elif hasattr(comfy.model_base, "BaseModel"):
            if info["context_dim"] == 768:
                info["type"] = "sd1"
            elif info["context_dim"] == 1024:
                info["type"] = "sd2"

        return info

    def _create_dummy_inputs(
        self,
        model_info: Dict[str, Any],
        batch: int,
        height: int,
        width: int,
        context_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], Dict]:
        """Create dummy inputs for ONNX export."""

        latent_height = height // 8
        latent_width = width // 8
        in_channels = model_info["in_channels"]
        context_dim = model_info["context_dim"]
        y_dim = model_info["y_dim"]

        dummy_inputs = {
            "x": torch.randn(batch, in_channels, latent_height, latent_width, device=device, dtype=dtype),
            "timesteps": torch.zeros(batch, device=device, dtype=dtype),
            "context": torch.randn(batch, context_len, context_dim, device=device, dtype=dtype),
        }

        input_names = ["x", "timesteps", "context"]
        output_names = ["output"]

        dynamic_axes = {
            "x": {0: "batch", 2: "height", 3: "width"},
            "timesteps": {0: "batch"},
            "context": {0: "batch", 1: "context_len"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

        if y_dim > 0:
            dummy_inputs["y"] = torch.randn(batch, y_dim, device=device, dtype=dtype)
            input_names.append("y")
            dynamic_axes["y"] = {0: "batch"}

        if model_info["type"] == "flux":
            dummy_inputs["guidance"] = torch.tensor([3.5], device=device, dtype=dtype)
            input_names.append("guidance")

        return dummy_inputs, input_names, output_names, dynamic_axes

    def _build_trt_engine(
        self,
        onnx_path: Path,
        engine_path: Path,
        input_names: List[str],
        model_info: Dict[str, Any],
        batch_min: int, batch_opt: int, batch_max: int,
        height_min: int, height_opt: int, height_max: int,
        width_min: int, width_opt: int, width_max: int,
        context_min: int, context_opt: int, context_max: int,
        quant_format: QuantFormat,
        dtype: torch.dtype,
        enable_refit: bool,
    ) -> None:
        """Build TensorRT engine from quantized ONNX model."""

        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)

        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

        if not parse_onnx_to_trt_network(onnx_path, builder, network):
            raise RuntimeError("Failed to parse ONNX model")

        # Create builder config with quantization flags
        config = builder.create_builder_config()

        # Set precision flags based on quantization
        if quant_format == QuantFormat.NVFP4:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.BF16)
            # TensorRT will use FP4 kernels from the quantized ONNX
            logger.info("Enabled FP16/BF16 flags for NVFP4 engine")
        elif quant_format == QuantFormat.FP8:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.FP8)
            logger.info("Enabled FP8 flag for engine")
        else:
            if dtype == torch.float16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif dtype == torch.bfloat16:
                config.set_flag(trt.BuilderFlag.BF16)

        # Enable refitting for LoRA support
        if enable_refit:
            config.set_flag(trt.BuilderFlag.REFIT)
            logger.info("Enabled REFIT flag for LoRA support")

        # Create optimization profile
        profile = builder.create_optimization_profile()

        in_channels = model_info["in_channels"]
        context_dim = model_info["context_dim"]
        y_dim = model_info["y_dim"]

        profile.set_shape(
            "x",
            (batch_min, in_channels, height_min // 8, width_min // 8),
            (batch_opt, in_channels, height_opt // 8, width_opt // 8),
            (batch_max, in_channels, height_max // 8, width_max // 8),
        )

        profile.set_shape("timesteps", (batch_min,), (batch_opt,), (batch_max,))

        profile.set_shape(
            "context",
            (batch_min, context_min, context_dim),
            (batch_opt, context_opt, context_dim),
            (batch_max, context_max, context_dim),
        )

        if y_dim > 0 and "y" in input_names:
            profile.set_shape("y", (batch_min, y_dim), (batch_opt, y_dim), (batch_max, y_dim))

        if "guidance" in input_names:
            profile.set_shape("guidance", (1,), (1,), (1,))

        config.add_optimization_profile(profile)

        # Build and serialize
        success = serialize_engine(builder, network, config, engine_path)
        save_timing_cache(config)

        if not success:
            raise RuntimeError("Failed to build TensorRT engine")

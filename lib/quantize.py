"""
Quantization utilities using NVIDIA Model Optimizer for FP4/FP8 quantization.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantFormat(Enum):
    """Supported quantization formats."""
    FP16 = "fp16"
    FP8 = "fp8"
    NVFP4 = "nvfp4"


@dataclass
class QuantConfig:
    """Configuration for model quantization."""
    format: QuantFormat
    calibration_size: int = 512
    calibration_batch_size: int = 1
    # Layers to skip quantization (embeddings, final output)
    skip_layers: Tuple[str, ...] = ("time_embed", "label_emb", "out.", "final_layer")


def check_quantization_available() -> Dict[str, bool]:
    """
    Check which quantization features are available.

    Returns:
        Dict with availability of each feature
    """
    available = {
        "modelopt": False,
        "fp8": False,
        "nvfp4": False,
    }

    # Check for nvidia-modelopt
    try:
        import modelopt  # noqa: F401
        available["modelopt"] = True
        logger.info("nvidia-modelopt is available")
    except ImportError:
        logger.warning(
            "nvidia-modelopt not installed. Install with: "
            "pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com"
        )
        return available

    # Check GPU capability for FP8/FP4
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        sm_version = capability[0] * 10 + capability[1]

        # FP8 requires SM >= 89 (Ada Lovelace)
        if sm_version >= 89:
            available["fp8"] = True
            logger.info(f"FP8 supported (SM {sm_version})")

        # NVFP4 requires SM >= 100 (Blackwell)
        if sm_version >= 100:
            available["nvfp4"] = True
            logger.info(f"NVFP4 supported (SM {sm_version})")

    return available


def get_calibration_data(
    model_type: str,
    batch_size: int,
    height: int,
    width: int,
    context_dim: int,
    context_len: int,
    device: torch.device,
    dtype: torch.dtype,
    num_samples: int = 512,
) -> Dict[str, torch.Tensor]:
    """
    Generate calibration data for PTQ.

    For diffusion models, we generate random latents and embeddings
    that cover the expected input distribution.
    """
    latent_height = height // 8
    latent_width = width // 8

    # Generate random calibration samples
    calibration_data = {
        "x": torch.randn(
            num_samples, 4, latent_height, latent_width,
            device=device, dtype=dtype
        ),
        "timesteps": torch.randint(
            0, 1000, (num_samples,),
            device=device, dtype=dtype
        ),
        "context": torch.randn(
            num_samples, context_len, context_dim,
            device=device, dtype=dtype
        ),
    }

    return calibration_data


def quantize_unet_fp8(
    unet: nn.Module,
    calibration_data: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[QuantConfig] = None,
) -> nn.Module:
    """
    Quantize UNet to FP8 using NVIDIA Model Optimizer.

    Args:
        unet: The UNet model to quantize
        calibration_data: Optional calibration data for PTQ
        config: Quantization configuration

    Returns:
        Quantized UNet model
    """
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.utils import set_quantizer_attribute
    except ImportError:
        raise ImportError(
            "nvidia-modelopt required for FP8 quantization. "
            "Install with: pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com"
        )

    config = config or QuantConfig(format=QuantFormat.FP8)

    logger.info("Quantizing UNet to FP8...")

    # Define quantization config
    quant_config = mtq.FP8_DEFAULT_CFG.copy()

    # Apply quantization
    def forward_loop(model):
        """Calibration forward loop."""
        if calibration_data is None:
            return

        model.eval()
        with torch.no_grad():
            batch_size = config.calibration_batch_size
            for i in range(0, min(len(calibration_data["x"]), config.calibration_size), batch_size):
                x = calibration_data["x"][i:i+batch_size]
                t = calibration_data["timesteps"][i:i+batch_size]
                c = calibration_data["context"][i:i+batch_size]

                try:
                    model(x, t, c)
                except Exception as e:
                    logger.debug(f"Calibration forward error (may be expected): {e}")
                    break

    # Quantize the model
    mtq.quantize(unet, quant_config, forward_loop)

    # Disable quantization for specified layers
    for name, module in unet.named_modules():
        if any(skip in name for skip in config.skip_layers):
            set_quantizer_attribute(module, "disable", True)
            logger.debug(f"Disabled quantization for: {name}")

    logger.info("FP8 quantization complete")
    return unet


def quantize_unet_nvfp4(
    unet: nn.Module,
    calibration_data: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[QuantConfig] = None,
) -> nn.Module:
    """
    Quantize UNet to NVFP4 using NVIDIA Model Optimizer.

    NVFP4 uses block quantization with 16-element blocks and FP8 scales.
    Requires Blackwell GPU (SM >= 10.0).

    Args:
        unet: The UNet model to quantize
        calibration_data: Optional calibration data for PTQ
        config: Quantization configuration

    Returns:
        Quantized UNet model
    """
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.utils import set_quantizer_attribute
    except ImportError:
        raise ImportError(
            "nvidia-modelopt required for NVFP4 quantization. "
            "Install with: pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com"
        )

    # Check GPU capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        sm_version = capability[0] * 10 + capability[1]
        if sm_version < 100:
            raise RuntimeError(
                f"NVFP4 requires Blackwell GPU (SM >= 10.0), "
                f"but current GPU is SM {sm_version}"
            )

    config = config or QuantConfig(format=QuantFormat.NVFP4)

    logger.info("Quantizing UNet to NVFP4...")

    # Define NVFP4 quantization config
    # NVFP4 uses E2M1 format with block size 16 and FP8 E4M3 scales
    quant_config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": 4,
                "block_sizes": {-1: 16},  # Block size of 16 along last dimension
                "enable": True,
            },
            "*input_quantizer": {
                "num_bits": 4,
                "block_sizes": {-1: 16},
                "enable": True,
            },
        },
        "algorithm": "max",
    }

    # Calibration forward loop
    def forward_loop(model):
        if calibration_data is None:
            return

        model.eval()
        with torch.no_grad():
            batch_size = config.calibration_batch_size
            for i in range(0, min(len(calibration_data["x"]), config.calibration_size), batch_size):
                x = calibration_data["x"][i:i+batch_size]
                t = calibration_data["timesteps"][i:i+batch_size]
                c = calibration_data["context"][i:i+batch_size]

                try:
                    model(x, t, c)
                except Exception as e:
                    logger.debug(f"Calibration forward error: {e}")
                    break

    # Apply quantization
    mtq.quantize(unet, quant_config, forward_loop)

    # Disable quantization for specified layers
    for name, module in unet.named_modules():
        if any(skip in name for skip in config.skip_layers):
            set_quantizer_attribute(module, "disable", True)
            logger.debug(f"Disabled quantization for: {name}")

    logger.info("NVFP4 quantization complete")
    return unet


def quantize_unet(
    unet: nn.Module,
    quant_format: QuantFormat,
    calibration_data: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[QuantConfig] = None,
) -> nn.Module:
    """
    Quantize UNet to specified format.

    Args:
        unet: The UNet model to quantize
        quant_format: Target quantization format (FP8 or NVFP4)
        calibration_data: Optional calibration data for PTQ
        config: Quantization configuration

    Returns:
        Quantized UNet model
    """
    if quant_format == QuantFormat.FP16:
        logger.info("FP16 selected - no quantization needed")
        return unet
    elif quant_format == QuantFormat.FP8:
        return quantize_unet_fp8(unet, calibration_data, config)
    elif quant_format == QuantFormat.NVFP4:
        return quantize_unet_nvfp4(unet, calibration_data, config)
    else:
        raise ValueError(f"Unknown quantization format: {quant_format}")


def export_quantized_onnx(
    model: nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    output_path: Path,
    input_names: list,
    output_names: list,
    dynamic_axes: Dict[str, Dict[int, str]],
    opset_version: int = 23,  # Required for FP4 support
) -> None:
    """
    Export a quantized model to ONNX format.

    For FP4/FP8 models, uses ONNX opset 23 which supports
    FP4E2M1 type and block quantization.

    Args:
        model: Quantized PyTorch model
        dummy_inputs: Dict of dummy input tensors
        output_path: Path to save ONNX file
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axis specifications
        opset_version: ONNX opset version (23 for FP4)
    """
    try:
        import modelopt.torch.opt as mto
    except ImportError:
        # Fall back to standard export if modelopt not available
        logger.warning("modelopt not available, using standard ONNX export")
        torch.onnx.export(
            model,
            tuple(dummy_inputs[name] for name in input_names),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        return

    logger.info(f"Exporting quantized model to ONNX (opset {opset_version})...")

    # Prepare inputs as tuple
    inputs = tuple(dummy_inputs[name] for name in input_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use modelopt export for proper quantized operator handling
    mto.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    logger.info(f"Quantized ONNX export complete: {output_path}")

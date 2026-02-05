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
            device=device, dtype=torch.float32  # timesteps should be float32, not bf16
        ).to(dtype),
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
        import modelopt.torch.opt as mto
    except ImportError as e:
        raise ImportError(
            f"nvidia-modelopt required for FP8 quantization. "
            f"Install with: pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com\n"
            f"Original error: {e}"
        )

    config = config or QuantConfig(format=QuantFormat.FP8)

    logger.info("Quantizing UNet to FP8...")

    # Log module types before quantization for debugging
    module_types = {}
    for name, module in unet.named_modules():
        module_type = type(module).__name__
        module_types[module_type] = module_types.get(module_type, 0) + 1
    logger.info(f"Model module types: {dict(sorted(module_types.items(), key=lambda x: -x[1])[:10])}")

    # Count Linear and Conv2d modules
    linear_count = sum(1 for m in unet.modules() if isinstance(m, nn.Linear))
    conv_count = sum(1 for m in unet.modules() if isinstance(m, nn.Conv2d))
    logger.info(f"Found {linear_count} Linear and {conv_count} Conv2d modules")

    # Calibration forward loop (no model parameter - uses closure)
    def forward_loop():
        """Calibration forward loop."""
        if calibration_data is None:
            return

        unet.eval()
        with torch.no_grad():
            batch_size = config.calibration_batch_size
            for i in range(0, min(len(calibration_data["x"]), config.calibration_size), batch_size):
                x = calibration_data["x"][i:i+batch_size]
                t = calibration_data["timesteps"][i:i+batch_size]
                c = calibration_data["context"][i:i+batch_size]

                try:
                    unet(x, t, c)
                except Exception as e:
                    logger.debug(f"Calibration forward error (may be expected): {e}")
                    break

    # Use the HuggingFace diffusers pattern:
    # 1. Apply quantization mode to insert quantizer modules
    # 2. Calibrate to collect statistics
    # 3. Compress to finalize quantization

    # Build config for FP8 quantization
    quant_config = {
        "quant_cfg": mtq.FP8_DEFAULT_CFG["quant_cfg"],
        "algorithm": "max",
    }

    logger.info("Step 1: Applying quantization mode (inserting quantizers)...")
    mto.apply_mode(unet, mode=[("quantize", quant_config)])

    # Check if quantizers were inserted
    quantizer_count = sum(1 for name, module in unet.named_modules()
                         if 'quantizer' in type(module).__name__.lower())
    logger.info(f"Quantizer modules inserted: {quantizer_count}")

    if quantizer_count > 0:
        logger.info("Step 2: Running calibration...")
        mtq.calibrate(unet, algorithm="max", forward_loop=forward_loop)

        logger.info("Step 3: Compressing model...")
        mtq.compress(unet)
    else:
        logger.warning("No quantizers inserted - falling back to mtq.quantize()...")
        # Fall back to direct quantize call
        mtq.quantize(unet, mtq.FP8_DEFAULT_CFG, lambda m: forward_loop())

    # Print quantization summary if available
    try:
        mtq.print_quant_summary(unet)
    except Exception:
        pass

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

    Note: NVFP4 block quantization only supports Linear layers.
    Conv2d layers remain in FP16/BF16 for accuracy (per NVIDIA guidance).
    This is the recommended pattern from NVIDIA for diffusion models.

    Args:
        unet: The UNet model to quantize
        calibration_data: Optional calibration data for PTQ
        config: Quantization configuration

    Returns:
        Quantized UNet model
    """
    try:
        import modelopt.torch.quantization as mtq
        import modelopt.torch.opt as mto
    except ImportError as e:
        raise ImportError(
            f"nvidia-modelopt required for NVFP4 quantization. "
            f"Install with: pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com\n"
            f"Original error: {e}"
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
    logger.info("Note: NVFP4 only quantizes Linear layers. Conv2d remains in FP16/BF16.")

    # Log module types before quantization for debugging
    module_types = {}
    for name, module in unet.named_modules():
        module_type = type(module).__name__
        module_types[module_type] = module_types.get(module_type, 0) + 1
    logger.info(f"Model module types: {dict(sorted(module_types.items(), key=lambda x: -x[1])[:10])}")

    # Count Linear and Conv2d modules
    linear_count = sum(1 for m in unet.modules() if isinstance(m, nn.Linear))
    conv_count = sum(1 for m in unet.modules() if isinstance(m, nn.Conv2d))
    logger.info(f"Found {linear_count} Linear and {conv_count} Conv2d modules")
    logger.info(f"NVFP4 will quantize {linear_count} Linear layers (Conv2d stays FP16/BF16)")

    # Calibration forward loop
    def forward_loop():
        if calibration_data is None:
            return

        unet.eval()
        with torch.no_grad():
            batch_size = config.calibration_batch_size
            for i in range(0, min(len(calibration_data["x"]), config.calibration_size), batch_size):
                x = calibration_data["x"][i:i+batch_size]
                t = calibration_data["timesteps"][i:i+batch_size]
                c = calibration_data["context"][i:i+batch_size]

                try:
                    unet(x, t, c)
                except Exception as e:
                    logger.debug(f"Calibration forward error: {e}")
                    break

    # Use the HuggingFace diffusers pattern:
    # 1. Apply quantization mode to insert quantizer modules
    # 2. Calibrate to collect statistics
    # 3. Compress to finalize quantization

    # Build config for NVFP4 quantization
    quant_config = {
        "quant_cfg": mtq.NVFP4_DEFAULT_CFG["quant_cfg"],
        "algorithm": "max",
    }

    logger.info("Step 1: Applying quantization mode (inserting quantizers)...")
    mto.apply_mode(unet, mode=[("quantize", quant_config)])

    # Check if quantizers were inserted
    quantizer_count = sum(1 for name, module in unet.named_modules()
                         if 'quantizer' in type(module).__name__.lower())
    logger.info(f"Quantizer modules inserted: {quantizer_count}")

    if quantizer_count > 0:
        logger.info("Step 2: Running calibration...")
        mtq.calibrate(unet, algorithm="max", forward_loop=forward_loop)

        logger.info("Step 3: Compressing model...")
        mtq.compress(unet)
    else:
        logger.warning("No quantizers inserted - falling back to mtq.quantize()...")
        # Fall back to direct quantize call
        mtq.quantize(unet, mtq.NVFP4_DEFAULT_CFG, lambda m: forward_loop())

    # Print quantization summary if available
    try:
        mtq.print_quant_summary(unet)
    except Exception:
        pass

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
    opset_version: int = 21,  # Opset 21 supports INT4, opset 23 for FP4E2M1
) -> None:
    """
    Export a quantized model to ONNX format.

    For FP4/FP8 models, uses higher ONNX opset versions which support
    quantized types and block quantization.

    Args:
        model: Quantized PyTorch model
        dummy_inputs: Dict of dummy input tensors
        output_path: Path to save ONNX file
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axis specifications
        opset_version: ONNX opset version
    """
    logger.info(f"Exporting quantized model to ONNX (opset {opset_version})...")

    # Prepare inputs as tuple
    inputs = tuple(dummy_inputs[name] for name in input_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use standard PyTorch ONNX export
    # The quantized model will include Q/DQ nodes in the graph
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

    logger.info(f"Quantized ONNX export complete: {output_path}")

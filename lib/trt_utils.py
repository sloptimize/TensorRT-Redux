"""
TensorRT utility functions for engine building and management.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import tensorrt as trt
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TIMING_CACHE = "timing_cache.trt"


class TQDMProgressMonitor(trt.IProgressMonitor):
    """Progress monitor with TQDM progress bars for TensorRT builds."""

    def __init__(self):
        super().__init__()
        self._active_phases: Dict[str, tqdm] = {}
        self._phase_stack: List[str] = []

    def phase_start(self, phase_name: str, parent_phase: Optional[str], num_steps: int) -> None:
        self._phase_stack.append(phase_name)
        indent = "  " * (len(self._phase_stack) - 1)

        if num_steps > 0:
            self._active_phases[phase_name] = tqdm(
                total=num_steps,
                desc=f"{indent}{phase_name}",
                leave=False,
                dynamic_ncols=True
            )
        else:
            logger.info(f"{indent}Starting: {phase_name}")

    def phase_finish(self, phase_name: str) -> None:
        if phase_name in self._active_phases:
            self._active_phases[phase_name].close()
            del self._active_phases[phase_name]

        if self._phase_stack and self._phase_stack[-1] == phase_name:
            self._phase_stack.pop()

    def step_complete(self, phase_name: str, step: int) -> bool:
        if phase_name in self._active_phases:
            pbar = self._active_phases[phase_name]
            pbar.update(1)
        return True  # Continue building


def get_timing_cache_path(cache_dir: Optional[str] = None) -> Path:
    """Get the path to the timing cache file."""
    if cache_dir:
        return Path(cache_dir) / DEFAULT_TIMING_CACHE
    return Path(DEFAULT_TIMING_CACHE)


def load_timing_cache(config: trt.IBuilderConfig, cache_path: Optional[Path] = None) -> None:
    """Load timing cache from file if it exists."""
    cache_path = cache_path or get_timing_cache_path()

    if cache_path.exists():
        logger.info(f"Loading timing cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache_data = f.read()

        timing_cache = config.create_timing_cache(cache_data)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)
    else:
        logger.info("No existing timing cache found, creating new one")
        timing_cache = config.create_timing_cache(b"")
        config.set_timing_cache(timing_cache, ignore_mismatch=True)


def save_timing_cache(config: trt.IBuilderConfig, cache_path: Optional[Path] = None) -> None:
    """Save timing cache to file for future builds."""
    cache_path = cache_path or get_timing_cache_path()

    timing_cache = config.get_timing_cache()
    if timing_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(timing_cache.serialize())
        logger.info(f"Saved timing cache to {cache_path}")


def create_builder_config(
    builder: trt.Builder,
    dtype: torch.dtype = torch.float16,
    enable_refit: bool = True,
    timing_cache_path: Optional[Path] = None,
    show_progress: bool = True
) -> trt.IBuilderConfig:
    """
    Create a TensorRT builder configuration with appropriate flags.

    Args:
        builder: TensorRT builder instance
        dtype: Target precision (torch.float16, torch.bfloat16)
        enable_refit: Enable weight refitting (required for LoRA support)
        timing_cache_path: Path to timing cache file
        show_progress: Show progress bars during build

    Returns:
        Configured IBuilderConfig
    """
    config = builder.create_builder_config()

    # Set precision flags
    if dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif dtype == torch.bfloat16:
        config.set_flag(trt.BuilderFlag.BF16)

    # Enable refitting for LoRA support
    if enable_refit:
        config.set_flag(trt.BuilderFlag.REFIT)
        logger.info("Enabled REFIT flag for LoRA support")

    # Load timing cache
    load_timing_cache(config, timing_cache_path)

    # Set progress monitor
    if show_progress:
        config.progress_monitor = TQDMProgressMonitor()

    return config


def create_optimization_profile(
    builder: trt.Builder,
    input_specs: Dict[str, Dict[str, Tuple[int, ...]]],
) -> trt.IOptimizationProfile:
    """
    Create an optimization profile with min/opt/max shapes.

    Args:
        builder: TensorRT builder instance
        input_specs: Dict mapping input names to shape specs:
            {
                "input_name": {
                    "min": (batch, channels, height, width),
                    "opt": (batch, channels, height, width),
                    "max": (batch, channels, height, width),
                }
            }

    Returns:
        Configured optimization profile
    """
    profile = builder.create_optimization_profile()

    for name, shapes in input_specs.items():
        profile.set_shape(
            name,
            min=shapes["min"],
            opt=shapes["opt"],
            max=shapes["max"]
        )
        logger.debug(f"Profile {name}: min={shapes['min']}, opt={shapes['opt']}, max={shapes['max']}")

    return profile


def serialize_engine(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    config: trt.IBuilderConfig,
    output_path: Path
) -> bool:
    """
    Build and serialize a TensorRT engine to file.

    Args:
        builder: TensorRT builder
        network: Network definition (from ONNX)
        config: Builder configuration
        output_path: Path to save .engine file

    Returns:
        True if successful
    """
    logger.info("Building TensorRT engine (this may take several minutes)...")

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"Saved TensorRT engine to {output_path}")
    return True


def load_engine(engine_path: Path) -> Tuple[trt.ICudaEngine, trt.IExecutionContext]:
    """
    Load a TensorRT engine from file.

    Args:
        engine_path: Path to .engine file

    Returns:
        Tuple of (engine, execution_context)
    """
    logger.info(f"Loading TensorRT engine from {engine_path}")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

    context = engine.create_execution_context()

    return engine, context


def check_refit_support(engine: trt.ICudaEngine) -> bool:
    """Check if an engine was built with REFIT flag enabled."""
    return engine.refittable


def get_engine_io_info(engine: trt.ICudaEngine) -> Dict[str, Dict[str, Any]]:
    """
    Get information about engine inputs and outputs.

    Returns:
        Dict with input/output names, shapes, and dtypes
    """
    info = {"inputs": {}, "outputs": {}}

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)

        tensor_info = {
            "shape": tuple(shape),
            "dtype": str(dtype),
        }

        if mode == trt.TensorIOMode.INPUT:
            info["inputs"][name] = tensor_info
        else:
            info["outputs"][name] = tensor_info

    return info


def trt_dtype_to_torch(trt_dtype: trt.DataType) -> torch.dtype:
    """Convert TensorRT dtype to PyTorch dtype."""
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
    }
    return mapping.get(trt_dtype, torch.float32)


def torch_dtype_to_trt(torch_dtype: torch.dtype) -> trt.DataType:
    """Convert PyTorch dtype to TensorRT dtype."""
    mapping = {
        torch.float32: trt.DataType.FLOAT,
        torch.float16: trt.DataType.HALF,
        torch.bfloat16: trt.DataType.BF16,
        torch.int32: trt.DataType.INT32,
        torch.int8: trt.DataType.INT8,
    }
    return mapping.get(torch_dtype, trt.DataType.FLOAT)

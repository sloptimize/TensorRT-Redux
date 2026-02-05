# TensorRT-Redux

**LoRA support for TensorRT-accelerated Stable Diffusion in ComfyUI**

TensorRT-Redux enables using LoRA models with TensorRT-compiled diffusion models by leveraging TensorRT's engine refitting capability. This allows weight updates at runtime without expensive recompilation.

## Features

- **LoRA Support**: Use your favorite LoRAs with TensorRT acceleration
- **No Recompilation**: Apply LoRAs via weight refitting (~1 second vs 5-30 minutes)
- **Multi-LoRA Stacking**: Combine multiple LoRAs with adjustable scales
- **SDXL Support**: Full support for SDXL-based models (more architectures coming)
- **Dynamic Resolution**: Support for variable batch sizes and resolutions

## How It Works

Traditional TensorRT engines have "baked-in" weights that can't be modified. TensorRT-Redux uses TensorRT's **refitting API** to update weights at runtime:

1. **Export Phase**: Build TensorRT engine with `REFIT` flag enabled, preserving the ONNX file
2. **LoRA Export**: Pre-compute weight deltas between base model and LoRA-applied model
3. **Runtime**: Load engine, apply deltas via `engine.refit()` - no recompilation needed

## Installation

### Requirements

- NVIDIA GPU with TensorRT support (RTX 20 series or newer recommended)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- TensorRT 10.0.1 or newer
- CUDA 12.x

### Install

```bash
# Clone into ComfyUI custom_nodes
cd ComfyUI/custom_nodes
git clone https://github.com/sloptimize/TensorRT-Redux.git

# Install base dependencies
pip install onnx safetensors numpy tqdm
```

### TensorRT Installation

#### Linux
```bash
pip install tensorrt>=10.0.1
```

#### Windows

TensorRT doesn't install cleanly via pip on Windows. Choose one option:

**Option 1: NVIDIA PyPI Index (easiest)**
```bash
# For CUDA 12.x
pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com

# For CUDA 13.x (Blackwell GPUs)
pip install tensorrt-cu13 --extra-index-url https://pypi.nvidia.com
```

**Option 2: Manual Download (most reliable)**
1. Download TensorRT 10.x from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
2. Extract the zip file
3. Install the wheel:
```bash
pip install <path_to_tensorrt>\python\tensorrt-10.x.x-cp3xx-none-win_amd64.whl
```
4. Add the `lib` folder to your PATH or copy DLLs to your Python environment

## Quick Start

### Step 1: Export Base Model to TensorRT

Use the **TensorRT Model Export (LoRA-Ready)** node:

1. Load your SDXL checkpoint
2. Connect to `TRT_MODEL_EXPORT` node
3. Configure resolution range (e.g., 512-1536)
4. Run to export `.engine`, `.onnx`, and `_weights.json` files

This takes 5-30 minutes depending on your GPU and resolution range.

### Step 2: Export LoRA Deltas

Use the **TensorRT LoRA Export** node:

1. Select your base engine from Step 1
2. Select your LoRA file
3. Set strength (default 1.0)
4. Run to generate `.trt_lora` delta file

This takes ~20 seconds per LoRA.

### Step 3: Load and Generate

Use the **TensorRT Loader + LoRA** node:

1. Select your engine
2. Select model type (sdxl_base, etc.)
3. Select your LoRA delta file
4. Adjust scale as needed
5. Connect to your workflow and generate!

## Nodes Reference

### TRT_MODEL_EXPORT
Export a ComfyUI model to TensorRT with refitting enabled.

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | ComfyUI model from Load Checkpoint |
| filename_prefix | STRING | Output path/name |
| batch_min/opt/max | INT | Batch size range |
| height_min/opt/max | INT | Height range (pixels) |
| width_min/opt/max | INT | Width range (pixels) |
| context_min/opt/max | INT | Context length range |
| enable_refit | BOOL | Enable LoRA support (default: True) |
| precision | ENUM | fp16 or bf16 |

**Outputs**: Saves `.engine`, `.onnx`, `_weights.json` to `models/tensorrt/`

### TRT_LORA_EXPORT
Pre-compute LoRA weight deltas for TensorRT refitting.

| Input | Type | Description |
|-------|------|-------------|
| base_engine | ENUM | Select base TensorRT engine |
| lora_file | ENUM | Select LoRA file |
| output_name | STRING | Name for output delta file |
| lora_strength | FLOAT | LoRA strength (default: 1.0) |

**Outputs**: Saves `.trt_lora` to `models/trt_lora/`

### TRT_LOADER_LORA
Load TensorRT engine with optional LoRA.

| Input | Type | Description |
|-------|------|-------------|
| engine_name | ENUM | Select TensorRT engine |
| model_type | ENUM | Model architecture |
| lora_delta | ENUM | Select LoRA delta file (optional) |
| lora_scale | FLOAT | Runtime scale adjustment |

**Outputs**: MODEL (ComfyUI model)

### TRT_LORA_STACK
Stack multiple LoRAs for combined application.

| Input | Type | Description |
|-------|------|-------------|
| lora_delta | ENUM | LoRA delta file |
| scale | FLOAT | Scale for this LoRA |
| lora_stack | TRT_LORA_STACK | Previous stack (optional) |

### TRT_LOADER_LORA_STACKED
Load TensorRT engine with multiple stacked LoRAs.

| Input | Type | Description |
|-------|------|-------------|
| engine_name | ENUM | Select TensorRT engine |
| model_type | ENUM | Model architecture |
| lora_stack | TRT_LORA_STACK | LoRA stack from TRT_LORA_STACK |

### TRT_MODEL_EXPORT_QUANTIZED
Export a quantized model to TensorRT with FP8 or NVFP4 precision.

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | ComfyUI model from Load Checkpoint |
| filename_prefix | STRING | Output path/name |
| quantization | ENUM | nvfp4, fp8, or fp16 |
| batch_min/opt/max | INT | Batch size range |
| height_min/opt/max | INT | Height range (pixels) |
| width_min/opt/max | INT | Width range (pixels) |
| context_min/opt/max | INT | Context length range |
| enable_refit | BOOL | Enable LoRA support (default: True) |
| calibration_steps | INT | PTQ calibration samples (default: 512) |

**Requirements**:
- NVFP4: RTX 50 series (Blackwell) + CUDA 13.0 + nvidia-modelopt
- FP8: RTX 40 series+ + nvidia-modelopt

**Install nvidia-modelopt**:
```bash
pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com
```

## Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| SDXL Base | ✅ Supported | Fully tested |
| SDXL Refiner | ✅ Supported | |
| SD 1.5 | 🔄 Planned | Phase 2 |
| SD 2.x | 🔄 Planned | Phase 2 |
| SD3 | 🔄 Planned | Phase 2 |
| Flux | 🔄 Planned | Phase 2 |
| Wan 2.2 | 🔄 Planned | Phase 2 |

## Quantization Support

| Precision | Hardware | Status | Speedup | VRAM Savings |
|-----------|----------|--------|---------|--------------|
| FP16 | All NVIDIA GPUs | ✅ Supported | ~2x | baseline |
| FP8 | RTX 40 series+ (Ada) | ✅ Supported | ~2.5x | ~40% |
| NVFP4 | RTX 50 series (Blackwell) | ✅ Supported | ~3x | ~60% |

### Using Quantized Export

1. Install nvidia-modelopt:
```bash
pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com
```

2. **(Recommended)** Install cuDNN for faster calibration:
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - **Easiest method**: Copy DLLs directly into your ComfyUI folder:
     ```
     copy "C:\Program Files\NVIDIA\CUDNN\v9.x\bin\13.1\*.dll" "path\to\ComfyUI\"
     ```
   - Or add the bin folder to PATH via System Properties → Environment Variables (don't use `setx` - it can truncate PATH)
   - Without cuDNN, calibration falls back to CPU (slower but still works)

3. Use **TensorRT Quantized Export (FP8/NVFP4)** node instead of the standard export
4. Select your quantization level (nvfp4, fp8, or fp16)
5. The node will:
   - Export model to ONNX
   - Quantize ONNX using NVIDIA Model Optimizer
   - Build optimized TensorRT engine

### LoRA with Quantized Models

LoRAs remain in FP16 for accuracy, following [NVIDIA's recommended pattern](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/). The TensorRT refitting applies FP16 LoRA deltas to the quantized engine - you get the speed benefits of quantization plus full LoRA flexibility.

## Performance

Benchmarks from [NVIDIA's FLUX testing](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/) on RTX 5090:

| Configuration | FLUX.1-dev (30 steps) | vs FP16 |
|--------------|----------------------|---------|
| FP16 | 10,931ms | baseline |
| FP8 | 6,681ms | 1.6x faster |
| **NVFP4** | **3,853ms** | **2.8x faster** |

VRAM usage: 23.9GB (FP16) → 11.1GB (NVFP4) = **54% reduction**

LoRA refitting adds minimal overhead (~1 second when switching LoRAs).

## Troubleshooting

### "Engine was not built with REFIT flag"
Re-export your model using `TRT_MODEL_EXPORT` with `enable_refit=True`.

### "No weight changes detected"
The LoRA may not match the model architecture. Ensure you're using a LoRA trained for your base model (e.g., SDXL LoRA for SDXL engine).

### ONNX file not found
Ensure the `.onnx` file is in the same directory as the `.engine` file. Both are created by `TRT_MODEL_EXPORT`.

### Out of memory during export
- Reduce resolution range (smaller max height/width)
- Close other GPU applications
- Use a smaller batch_max value

### "cuDNN library not found" during quantization
This warning means calibration will use CPU (slower but works). To enable GPU calibration:
1. Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
2. Copy DLLs to your ComfyUI folder (easiest), or add to PATH via System Properties GUI:
   - Windows: `C:\Program Files\NVIDIA\CUDNN\v9.x\bin\13.1` (for CUDA 13)
   - Linux: `/usr/local/cuda/lib64` or where cuDNN is installed
   - **Warning**: Don't use `setx` to modify PATH - it can truncate long paths
3. Restart ComfyUI

## Technical Details

### TensorRT Refitting
TensorRT engines compiled with `BuilderFlag.REFIT` allow weight updates without recompilation. We use this to:

1. Store original ONNX weights for reference
2. Compute deltas: `delta = lora_modified_weight - original_weight`
3. Apply deltas at runtime: `engine.refit(weight_name, new_weight)`

### LoRA Delta Format
`.trt_lora` files use safetensors format containing:
- Weight deltas (only changed weights)
- Metadata (base engine, LoRA file, strength)

### Multi-LoRA Merging
Multiple LoRAs are merged before refitting:
```python
merged[name] = sum(delta[name] * scale for delta, scale in lora_stack)
```

## Contributing

Contributions welcome! Areas of interest:
- Additional model architecture support
- FP8/NVFP4 quantization
- ControlNet support
- Performance optimizations

## License

MIT License

## Acknowledgments

- [ComfyUI_TensorRT](https://github.com/comfyanonymous/ComfyUI_TensorRT) - Original TensorRT nodes
- [NVIDIA Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT) - LoRA refitting approach
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - Inference optimization
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) - FP8/NVFP4 quantization

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
git clone https://github.com/yourusername/TensorRT-Redux.git

# Install dependencies
pip install -r TensorRT-Redux/requirements.txt
```

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

## Quantization Roadmap

| Precision | Hardware | Status |
|-----------|----------|--------|
| FP16 | All NVIDIA GPUs | ✅ Supported |
| FP8 | RTX 40 series+ | 🔄 Phase 2 |
| NVFP4 | RTX 50 series (Blackwell) | 🔄 Phase 3 |

NVFP4 quantization can provide ~3x speedup with minimal quality loss. LoRAs remain in FP16 for accuracy (following [NVIDIA's recommended pattern](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)).

## Performance

Typical speedups vs standard PyTorch inference:

| Configuration | Speedup |
|--------------|---------|
| TensorRT FP16 | ~2x |
| TensorRT FP8 | ~2.5x |
| TensorRT NVFP4 | ~3x |

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

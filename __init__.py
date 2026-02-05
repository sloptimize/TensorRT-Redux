"""
TensorRT-Redux: LoRA Support for ComfyUI TensorRT

A ComfyUI custom node package that enables LoRA support with TensorRT-accelerated inference
using TensorRT's engine refitting capability.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.1.0"

WEB_DIRECTORY = "./web"

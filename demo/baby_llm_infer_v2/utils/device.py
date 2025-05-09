import torch
from typing import Optional, Union

def get_optimal_device(device: str = "auto") -> str:
    """Determine the optimal device for model inference
    
    Args:
        device: Device specification ("auto", "cuda", "cpu", "mps", etc.)
        
    Returns:
        String representing the device to use
    """
    if device != "auto":
        # Honor explicit device specification
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but not available")
        return device
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    """Get the appropriate torch dtype based on name and device
    
    Args:
        dtype_name: Name of the dtype ("float32", "float16", "bfloat16", "auto")
        device: Device being used
        
    Returns:
        The corresponding torch.dtype
    """
    if dtype_name == "float32" or dtype_name == "float":
        return torch.float32
    if dtype_name == "float16" or dtype_name == "half":
        return torch.float16
    if dtype_name == "bfloat16":
        if hasattr(torch, "bfloat16"):
            return torch.bfloat16
        raise ValueError("bfloat16 requested but not available in this PyTorch version")
    
    # Auto mode - choose based on device
    if dtype_name == "auto":
        if device == "cpu":
            return torch.float32
        else:
            return torch.float16
    
    raise ValueError(f"Unsupported dtype: {dtype_name}")
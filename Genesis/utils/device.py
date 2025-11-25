"""
Device and dtype utilities for Genesis Core

Author: eddy
Date: 2025-11-12
"""

import torch
from typing import Optional, Union


def get_device(device_str: str = "gpu") -> torch.device:
    """
    Get torch device from string specification

    Args:
        device_str: Device specification ("gpu", "cpu", "cuda", "offload_device")

    Returns:
        torch.device object
    """
    if device_str in ["gpu", "cuda"]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "offload_device":
        # For offload, use CPU as fallback
        return torch.device("cpu")
    else:
        # Try to parse as direct device string
        return torch.device(device_str)


def get_dtype(dtype_str: str = "fp16") -> torch.dtype:
    """
    Get torch dtype from string specification

    Args:
        dtype_str: Dtype specification ("fp16", "bf16", "fp32", etc.)

    Returns:
        torch.dtype object
    """
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    return dtype_map.get(dtype_str.lower(), torch.float32)


def get_optimal_device() -> torch.device:
    """
    Get the optimal available device

    Returns:
        torch.device object for best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get information about available devices

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_devices": [],
        "cpu_threads": torch.get_num_threads(),
        "optimal_device": str(get_optimal_device())
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "index": i,
                "name": device_props.name,
                "memory_total": device_props.total_memory,
                "memory_available": torch.cuda.mem_get_info(i)[0] if torch.cuda.is_available() else 0,
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            })

    return info


def move_to_device(tensor_or_model: Union[torch.Tensor, torch.nn.Module],
                   device: Optional[torch.device] = None) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move tensor or model to specified device

    Args:
        tensor_or_model: Tensor or model to move
        device: Target device (None for optimal device)

    Returns:
        Moved tensor or model
    """
    if device is None:
        device = get_optimal_device()

    return tensor_or_model.to(device)
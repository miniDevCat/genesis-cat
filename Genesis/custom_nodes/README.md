# Genesis Custom Nodes

This directory contains custom nodes for Genesis Engine.

## Required Nodes

### ComfyUI-WanVideoWrapper

The WanVideo nodes are required for video generation functionality.

**Installation:**

1. Clone the WanVideoWrapper repository:
```bash
cd genesis/custom_nodes/Comfyui
node:https://huggingface.co/eddy1111111/gift/blob/main/ComfyUI-WanVideoWrapper1101.rar
```

2. Or download from the official source and extract to:
```
genesis/custom_nodes/Comfyui/ComfyUI-WanVideoWrapper/
```

## Directory Structure

```
custom_nodes/
├── Comfyui/
│   └── ComfyUI-WanVideoWrapper/  # Place WanVideo nodes here
├── K_trans/                       # Optional: K_trans nodes
├── LLamacpp/                      # Optional: LLama nodes
└── WEBui/                         # Optional: Web UI nodes
```

## Note

Due to GitHub file size limitations, custom nodes are not included in this repository.
Please download them separately from their official sources.

---

**Author:** eddy
**Date:** 2025-11-12

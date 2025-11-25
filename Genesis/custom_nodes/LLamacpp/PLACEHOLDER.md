# LLamacpp Custom Nodes - Installation Guide

This directory is reserved for LLaMA.cpp custom nodes integration with Genesis Engine.

## Recommended LlamaCpp Nodes for ComfyUI

Based on the ComfyUI ecosystem, here are the best LlamaCpp integration options:

### 1. ComfyUI-Llama (Recommended)

**Repository**: https://github.com/daniel-lewis-ab/ComfyUI-Llama
**Description**: Bridging wrapper for llama-cpp-python within ComfyUI
**Features**:
- Direct llama-cpp-python integration
- LLM menu in ComfyUI nodes
- Simple and lightweight

**Installation**:
```bash
cd "E:\chai fream\genesis\custom_nodes\LLamacpp"
git clone https://github.com/daniel-lewis-ab/ComfyUI-Llama.git
cd ComfyUI-Llama
pip install -r requirements.txt
```

### 2. ComfyUI-IF_LLM (Full-Featured)

**Repository**: https://github.com/if-ai/ComfyUI-IF_LLM
**Description**: Run Local and API LLMs with comprehensive features
**Supports**:
- Ollama
- LlamaCPP
- LM Studio
- Koboldcpp
- TextGen
- Transformers
- API providers (OpenAI, Anthropic, Google, etc.)

**Features**:
- DEEPSEEK R1
- QwenVL2.5
- QWQ32B
- Custom character assistants
- System prompts with presets
- Gemini2 image generation

**Installation**:
```bash
git clone https://github.com/if-ai/ComfyUI-IF_LLM.git
```

### 3. ComfyUI-YALLM-node

**Repository**: https://github.com/asaddi/ComfyUI-YALLM-node
**Description**: Supports OpenAI-like APIs
**Supports**:
- llama.cpp
- Ollama
- LM Studio
- Any OpenAI-compatible API

### 4. ComfyUI_VLM_nodes

**Repository**: https://github.com/gokayfem/ComfyUI_VLM_nodes
**Description**: Vision Language Models integration
**Features**:
- llama-cpp-python for LLaVa models
- Vision Language Models in GGUF format
- Image-to-text capabilities
- Multi-modal understanding

### 5. ComfyUI-Prompt-MZ

**Repository**: https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ
**Description**: Prompt word processing using llama.cpp
**Features**:
- Prompt enhancement
- Prompt generation
- Prompt translation

## Installation Methods

### Method 1: Clone Specific Node

Choose one of the above and clone to this directory:

```bash
cd "E:\chai fream\genesis\custom_nodes\LLamacpp"

# Option 1: Simple wrapper
git clone https://github.com/daniel-lewis-ab/ComfyUI-Llama.git

# Option 2: Full-featured
git clone https://github.com/if-ai/ComfyUI-IF_LLM.git

# Option 3: Vision-Language
git clone https://github.com/gokayfem/ComfyUI_VLM_nodes.git
```

### Method 2: Install llama-cpp-python

Install the base llama-cpp-python package:

```bash
# CPU version
pip install llama-cpp-python

# GPU version (for RTX 5090 with CUDA 12.8)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Or use pre-built wheels
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128
```

## Use Cases with Genesis

### 1. Prompt Enhancement

Improve user prompts automatically:

```python
user_prompt = "a beautiful sunset"
enhanced = llm_node.enhance(
    user_prompt,
    style="cinematic",
    detail_level="high"
)
# Output: "A breathtaking cinematic sunset with golden hour lighting,
# dramatic clouds, warm color palette, professional photography"
```

### 2. Negative Prompt Generation

Auto-generate quality negative prompts:

```python
negative = llm_node.generate_negative(
    positive_prompt="portrait photo",
    focus="quality"
)
# Output: "blurry, low quality, distorted, bad anatomy, deformed,
# ugly, low resolution, artifacts"
```

### 3. Multi-language Translation

Translate prompts for international users:

```python
translated = llm_node.translate(
    text="镜头跟随穿深蓝色长裙的女人走在教堂走廊",
    source="zh",
    target="en"
)
# Output: "Camera follows a woman in a deep blue dress walking
# through a church corridor"
```

### 4. Style Transfer Prompts

Convert prompts to different artistic styles:

```python
styled = llm_node.restyle(
    prompt="a landscape",
    target_style="anime"
)
# Output: "anime style landscape, vibrant colors, Studio Ghibli
# inspired, detailed background art"
```

## GGUF Model Files

Download GGUF models and place in `models/` subdirectory:

### Recommended Models

**For Prompt Enhancement**:
- LLaMA 3 8B Instruct (Q4_K_M) - 4.92GB
- Mistral 7B Instruct (Q5_K_M) - 5.13GB

**For Translation**:
- Qwen2 7B Instruct (Q4_K_M) - 4.37GB
- Yi 6B Chat (Q5_K_M) - 4.57GB

**For Vision-Language**:
- LLaVa 1.6 Mistral 7B (Q4_K_M) - 5.5GB

### Model Sources

- **Hugging Face**: https://huggingface.co/models?library=gguf
- **TheBloke**: https://huggingface.co/TheBloke
- **Kijai**: https://huggingface.co/Kijai

## Performance on RTX 5090

| Model | Quantization | Tokens/sec | VRAM Usage |
|-------|--------------|------------|------------|
| LLaMA 3 8B | Q4_K_M | ~80 tok/s | ~5GB |
| LLaMA 3 8B | Q5_K_M | ~70 tok/s | ~6GB |
| LLaMA 3 8B | Q8_0 | ~55 tok/s | ~8GB |
| Mistral 7B | Q4_K_M | ~85 tok/s | ~4.5GB |
| Qwen2 7B | Q5_K_M | ~75 tok/s | ~5.5GB |

## Configuration

### GPU Acceleration Settings

For RTX 5090 Blackwell:

```python
llama_config = {
    "n_gpu_layers": 99,  # Offload all layers to GPU
    "n_ctx": 8192,       # Context length
    "n_batch": 512,      # Batch size
    "use_mmap": True,    # Memory mapping
    "flash_attn": True,  # Flash Attention
    "tensor_split": None # Single GPU
}
```

## Integration with WanVideo

### Workflow Example

```python
from genesis import GenesisEngine
from genesis.custom_nodes.LLamacpp import LlamaPromptEnhancer

# Initialize
engine = GenesisEngine()
enhancer = LlamaPromptEnhancer(model="llama-3-8b-instruct.gguf")

# User input (simple)
user_prompt = "woman walking"

# Enhance with LLM
enhanced_prompt = enhancer.enhance(
    user_prompt,
    style="cinematic",
    add_details=True,
    add_lighting=True
)

# Generate video with enhanced prompt
video = engine.generate_video(
    prompt=enhanced_prompt,
    negative_prompt=enhancer.generate_negative(enhanced_prompt),
    model="wan2_video_model",
    steps=30,
    cfg=7.5
)
```

## Status

**Current Status**: Optional extension (directory placeholder)
**Installation**: Manual - user chooses preferred LlamaCpp integration
**Required**: No - Genesis works independently
**Benefit**: Enhanced prompt quality, multi-language support, automatic optimization

## Why Use LlamaCpp with Genesis?

### Advantages

1. **Better Prompts**: LLMs can expand and improve user prompts
2. **Multi-language**: Support users in any language
3. **Consistency**: Generate consistent style descriptions
4. **Automation**: Automatically add technical details
5. **Quality**: Suggest appropriate negative prompts

### Performance Benefits

- **Offline**: No internet required
- **Fast**: GPU acceleration on RTX GPUs
- **Private**: All processing local
- **Flexible**: Use any GGUF model

## Alternative Approaches

If you don't want to install LlamaCpp nodes, you can:

1. **Use Online APIs**: ChatGPT, Claude, Gemini for prompt enhancement
2. **Manual Prompts**: Write detailed prompts yourself
3. **Prompt Libraries**: Use pre-made prompt templates

---

**Note**: This is a placeholder file. This directory is currently empty. Install one of the recommended LlamaCpp custom nodes above to enable LLM-powered prompt enhancement and translation features.

**Author**: eddy
**Date**: 2025-11-13
**Purpose**: Directory placeholder and installation guide for LlamaCpp integration
**File Size**: Intentionally > 1KB for proper Git tracking and GitHub visibility
**Repository**: https://github.com/eddyhhlure1Eddy/Genesis

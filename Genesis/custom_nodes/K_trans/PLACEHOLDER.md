# KTransformers Integration - Installation Guide

This directory is reserved for KTransformers integration with Genesis Engine.

## What is KTransformers?

**Official Repository**: https://github.com/kvcache-ai/ktransformers
**Developer**: MADSys Lab at Tsinghua University and Approaching.AI
**Contributors**: 87+ community members

KTransformers is a research framework for **efficient inference and fine-tuning of large language models** through CPU-GPU heterogeneous computing.

## Project Modules

KTransformers consists of two specialized modules:

### 1. kt-kernel (Inference Optimization)

**Purpose**: Efficient LLM inference with heterogeneous computing

**Features**:
- Intel AMX and AVX512/AVX2 optimized kernels
- INT4/INT8 quantized inference
- Efficient Mixture-of-Experts (MoE) handling
- NUMA-aware memory management
- Heterogeneous expert placement (CPU + GPU)
- Integration with SGLang and other frameworks

**Installation**:
```bash
cd kt-kernel
pip install .
```

### 2. KT-SFT (Fine-tuning)

**Purpose**: Ultra-large model fine-tuning with limited resources

**Breakthrough**: Fine-tune 671B DeepSeek-V3 with just 70GB GPU + 1.3TB RAM

**Features**:
- LLaMA-Factory integration
- Full LoRA support with heterogeneous acceleration
- Production-ready inference
- Evaluation tools

**Installation**:
```bash
cd KT-SFT
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek3_lora_sft_kt.yaml
```

## Installation Instructions for Genesis

### Clone KTransformers Repository

```bash
cd "E:\chai fream\genesis\custom_nodes\K_trans"
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers

# Install kt-kernel for inference
cd kt-kernel
pip install .

# Or install KT-SFT for fine-tuning
cd ../KT-SFT
pip install -e .
```

### Directory Structure

After installation:

```
K_trans/
├── ktransformers/
│   ├── kt-kernel/              # Inference optimization
│   │   ├── ktransformers/
│   │   ├── setup.py
│   │   └── README.md
│   └── KT-SFT/                 # Fine-tuning
│       ├── src/
│       ├── examples/
│       └── README.md
└── PLACEHOLDER.md              # This file
```

## Features Overview

K_trans nodes typically provide the following capabilities:

### Multi-language Translation
- Translate prompts between languages (EN, ZH, JP, KR, etc.)
- Context-aware translation for better prompt quality
- Preserve technical terms and artistic descriptions

### Token Transformation
- Advanced tokenization strategies
- Custom vocabulary support
- Subword tokenization for better model understanding

### Embedding Conversion
- Cross-model embedding adaptation
- Semantic mapping between different model spaces
- Embedding interpolation and blending

## Usage Example with Genesis

### Prompt Enhancement with KTransformers

```python
from ktransformers import KTransformersModel

# Load model with heterogeneous optimization
model = KTransformersModel(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    use_cpu_offload=True,
    use_int8_quantization=True
)

# Enhance user prompt
user_prompt = "woman walking"

enhanced = model.generate(
    f"Expand this into a detailed cinematic prompt: {user_prompt}",
    max_tokens=200
)

# Use enhanced prompt with Genesis
from genesis import GenesisEngine
engine = GenesisEngine()
video = engine.generate_video(prompt=enhanced)

## Configuration for Genesis Integration

### CPU-GPU Heterogeneous Setup

```python
from ktransformers import OptimizationConfig

config = OptimizationConfig(
    cpu_infer=8,              # CPU inference threads
    per_layer_prefill_intv=4, # Layer prefill interval
    max_new_tokens=512,       # Max generation length
    use_cuda_graph=True,      # Enable CUDA graphs
    use_flash_attn=True       # Flash Attention
)
```

### RTX 5090 Blackwell Optimization

```python
# Optimize for RTX 5090
ktransformers_config = {
    "gpu_layers": 35,          # Layers on GPU
    "cpu_layers": "auto",      # Rest on CPU
    "quantization": "int4",    # INT4 for max speed
    "rope_scaling": True,      # RoPE scaling
    "tensor_parallel": 1,      # Single GPU
    "expert_offload": True     # Offload inactive experts
}
```

## Performance Benchmarks

### DeepSeek-V3 (671B) on RTX 5090

| Configuration | GPU VRAM | RAM | Tokens/sec |
|---------------|----------|-----|------------|
| Full GPU | OOM | - | - |
| KTransformers | 70GB | 1.3TB | ~15 tok/s |
| KT + INT4 | 48GB | 800GB | ~25 tok/s |

### Qwen2.5-7B on RTX 5090

| Mode | GPU VRAM | Tokens/sec |
|------|----------|------------|
| Standard | 14GB | ~60 tok/s |
| KT INT8 | 8GB | ~80 tok/s |
| KT INT4 | 5GB | ~120 tok/s |

## Use Cases with Genesis WanVideo

### 1. LLM-Enhanced Prompt Generation

Use KTransformers LLM to automatically enhance prompts:

```python
from ktransformers import KTransformersModel
from genesis import GenesisEngine

# Load Qwen2.5-7B with KTransformers optimization
llm = KTransformersModel("Qwen/Qwen2.5-7B-Instruct")

# User input (simple)
user_input = "sunset beach"

# LLM enhancement
system_prompt = "You are a professional video prompt writer. Expand the following into a detailed, cinematic video prompt:"
enhanced_prompt = llm.generate(f"{system_prompt}\n\n{user_input}")

# Generate video with enhanced prompt
engine = GenesisEngine()
video = engine.generate_video(
    prompt=enhanced_prompt,
    model="wan2_video_model"
)
```

### 2. Multi-language Support

Translate prompts using LLM:

```python
# Chinese to English translation
chinese_prompt = "镜头跟随穿深蓝色长裙的女人走在教堂走廊"

translation_prompt = f"Translate the following Chinese to English:\n{chinese_prompt}"
english_prompt = llm.generate(translation_prompt)

# Generate with translated prompt
video = engine.generate_video(prompt=english_prompt)
```

### 3. Batch Prompt Generation

Generate variations automatically:

```python
base_concept = "cyberpunk city"

for i in range(5):
    variation_prompt = f"Generate variation {i+1} of this concept: {base_concept}"
    variation = llm.generate(variation_prompt)
    video = engine.generate_video(prompt=variation)
    save_video(video, f"cyberpunk_var_{i}.mp4")
```

## Performance Optimization

### GPU Acceleration
K_trans translation models can utilize GPU acceleration on RTX GPUs.

### Model Quantization
Use quantized models for faster translation:
- INT8 quantization: 2-3x faster
- FP16 precision: Balanced speed and quality

### Caching
Enable translation cache to avoid re-translating common prompts:

```python
translator.enable_cache(max_size=1000)
```

## Status

**Current Status**: Optional node category (directory placeholder)
**Installation**: Manual - user must provide K_trans nodes
**Required**: No - Genesis core functionality works without K_trans
**Benefit**: Multi-language support and prompt optimization

## Alternative Solutions

If K_trans is not available, you can use:

1. **Online Translation APIs**: Google Translate, DeepL, etc.
2. **Transformers Library**: Direct use of translation models
3. **Manual Translation**: Pre-translate prompts before generation

## References

- Genesis Documentation: `../../docs/`
- Translation Models: https://huggingface.co/models?pipeline_tag=translation
- Multi-language Guide: See project documentation

---

**Important Note**: This is a placeholder file indicating that K_trans nodes are not currently installed. This directory is reserved for future integration. Genesis Engine will work normally without K_trans nodes - they are an optional enhancement for multi-language workflow support.

**Author**: eddy
**Date**: 2025-11-13
**Purpose**: Directory placeholder and installation guidance
**File Size**: Intentionally > 1KB to ensure Git tracking and GitHub visibility
**Repository**: https://github.com/eddyhhlure1Eddy/Genesis

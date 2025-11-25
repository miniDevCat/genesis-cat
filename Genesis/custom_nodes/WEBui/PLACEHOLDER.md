# WEBui Custom Nodes - Installation Guide

This directory is reserved for Web UI related custom nodes and extensions.

## Overview

This directory can contain custom web UI components and nodes for Genesis Engine integration with various web frameworks.

## Relationship: ComfyUI WebUI Integration

According to the ComfyUI ecosystem, there's a bridge between Stable Diffusion WebUI and ComfyUI:

### sd-webui-comfyui Integration

**Repository**: https://github.com/ModelSurge/sd-webui-comfyui

This extension allows running ComfyUI workflows from Stable Diffusion WebUI. On startup, it scans WebUI extensions and injects enabled extension nodes into ComfyUI.

**Key Features**:
- Scan WebUI extensions for ComfyUI nodes
- Automatically inject nodes into ComfyUI
- Support `comfyui_custom_nodes/` path in extensions
- Bridge between WebUI and ComfyUI ecosystems

## Developing Custom Nodes from WebUI Extensions

When developing custom nodes that work with both systems:

### Directory Structure

```
WEBui/
├── __init__.py
├── comfyui_custom_nodes/     # ComfyUI nodes
│   ├── webui_bridge.py
│   ├── custom_controls.py
│   └── ...
├── scripts/                   # WebUI scripts
│   └── webui_script.py
└── javascript/               # Frontend JS
    └── custom_ui.js
```

### Node Discovery

sd-webui-comfyui scans for these paths:
1. `comfyui_custom_nodes/` - Python custom nodes
2. `javascript/` - Frontend components
3. `scripts/` - WebUI integration scripts

## Popular WebUI-Compatible Custom Nodes

### 1. WAS Node Suite

**Repository**: https://github.com/WASasquatch/comfyui-plugins
**Description**: Extensions, custom nodes, and plugins for ComfyUI
**Features**:
- 100+ custom nodes
- Image processing
- Text processing
- Math operations
- Video handling

### 2. ComfyUI-Custom-Nodes

**Repository**: https://github.com/Zuellni/ComfyUI-Custom-Nodes
**Description**: Various custom nodes for ComfyUI
**Categories**:
- Image manipulation
- Text generation
- File operations
- Utility nodes

### 3. ComfyUI_Custom_Nodes_AlekPet

**Repository**: https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet
**Description**: Custom nodes extending ComfyUI capabilities
**Features**:
- Advanced image processing
- Color adjustments
- Special effects
- UI enhancements

## Installation Methods

### Method 1: Install WebUI Bridge (For SD WebUI Users)

If you're using Stable Diffusion WebUI:

```bash
# Install sd-webui-comfyui extension
cd stable-diffusion-webui/extensions
git clone https://github.com/ModelSurge/sd-webui-comfyui.git
```

### Method 2: Install Standalone Custom Nodes

For Genesis standalone usage:

```bash
cd "E:\chai fream\genesis\custom_nodes\WEBui"

# Install WAS Node Suite
git clone https://github.com/WASasquatch/comfyui-plugins.git

# Or install other custom node packages
git clone https://github.com/Zuellni/ComfyUI-Custom-Nodes.git
```

### Method 3: Use ComfyUI Manager

ComfyUI Manager provides a GUI for installing custom nodes:

```bash
# Install ComfyUI Manager
cd "E:\chai fream\genesis\custom_nodes"
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

Then browse and install nodes through the UI.

## Use Cases with Genesis

### 1. Enhanced UI Controls

Add custom UI components to Genesis web interface:

```python
from genesis.custom_nodes.WEBui import CustomControls

controls = CustomControls()
controls.add_slider("CFG Scale", min=0, max=30, default=7.5)
controls.add_dropdown("Sampler", choices=["euler", "ddim", "unipc"])
controls.add_checkbox("Enable LoRA", default=True)
```

### 2. Batch Processing UI

Create batch processing interfaces:

```python
from genesis.custom_nodes.WEBui import BatchProcessor

processor = BatchProcessor()
processor.add_prompts_from_file("prompts.txt")
processor.set_output_directory("batch_output/")
processor.run_batch()
```

### 3. Image Gallery

Display generated videos/images in organized galleries:

```python
from genesis.custom_nodes.WEBui import Gallery

gallery = Gallery()
gallery.load_from_directory("genesis/output/")
gallery.display(columns=4, thumbnail_size=256)
```

### 4. Workflow Templates

Save and load workflow templates:

```python
from genesis.custom_nodes.WEBui import WorkflowManager

wf_manager = WorkflowManager()
wf_manager.save_template("my_video_workflow", current_settings)
wf_manager.load_template("my_video_workflow")
```

## Genesis Built-in Web UI

Genesis already includes a full-featured Gradio web interface:

### wanvideo_gradio_app.py

Located at: `E:\chai fream\genesis\apps\wanvideo_gradio_app.py`

**Features**:
- ✅ Text-to-video generation
- ✅ Model selection and management
- ✅ LoRA support
- ✅ Optimization controls (Torch Compile, Block Swap)
- ✅ Preset configurations
- ✅ Real-time progress tracking
- ✅ Frame preview gallery
- ✅ Metadata display

**No additional WEBui nodes needed for basic functionality!**

## When to Use Custom WEBui Nodes

Consider installing custom WebUI nodes if you need:

1. **Advanced Batch Processing**: Process hundreds of prompts
2. **Custom Workflows**: Non-standard generation pipelines
3. **Integration**: Connect Genesis with other tools
4. **Automation**: Scripted generation workflows
5. **Special UI**: Custom interface requirements

## Web Frameworks Supported

Genesis can integrate with:

- ✅ **Gradio** (built-in)
- ✅ **Streamlit**
- ✅ **Flask**
- ✅ **FastAPI**
- ✅ **Django**
- ✅ **Stable Diffusion WebUI** (via sd-webui-comfyui)

## Status

**Current Status**: Optional extension (directory placeholder)
**Installation**: Manual - user chooses preferred WebUI nodes
**Required**: No - Genesis includes built-in Gradio UI
**Benefit**: Extended UI capabilities and workflow automation

## Recommended Setup

For most users, the built-in Genesis Gradio app is sufficient. Install custom WEBui nodes only if you need specific advanced features not provided by the default interface.

## References

- **ComfyUI Custom Nodes List**: https://comfyui-wiki.com/en/resource/custom-nodes
- **ComfyUI Documentation**: https://docs.comfy.org/
- **Genesis Gradio App**: `genesis/apps/wanvideo_gradio_app.py`
- **sd-webui-comfyui**: https://github.com/ModelSurge/sd-webui-comfyui

---

**Note**: This is a placeholder file. This directory is currently empty and reserved for optional WebUI custom node integration. The Genesis Engine includes a complete Gradio-based web interface by default, so additional WebUI nodes are only needed for advanced customization scenarios.

**Author**: eddy
**Date**: 2025-11-13
**Purpose**: Directory placeholder for optional WebUI node extensions
**File Size**: Intentionally > 1KB for proper Git tracking and GitHub directory visibility
**Repository**: https://github.com/eddyhhlure1Eddy/Genesis

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis 简单演示界面 - 兼容版本
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    print("请安装 Gradio: pip install gradio")
    sys.exit(1)

# 导入 folder_paths
import importlib.util
spec = importlib.util.spec_from_file_location(
    "folder_paths", 
    Path(__file__).parent.parent / "core" / "folder_paths.py"
)
folder_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folder_paths)


def get_models():
    """获取模型列表"""
    return {
        'checkpoints': folder_paths.get_filename_list('checkpoints'),
        'loras': folder_paths.get_filename_list('loras'),
        'vaes': folder_paths.get_filename_list('vae')
    }


def show_models():
    """显示模型信息"""
    models = get_models()
    
    info = f"""
## 📦 可用模型

### Checkpoints: {len(models['checkpoints'])} 个
"""
    for i, name in enumerate(models['checkpoints'][:10], 1):
        info += f"{i}. {name}\n"
    if len(models['checkpoints']) > 10:
        info += f"... 还有 {len(models['checkpoints']) - 10} 个\n"
    
    info += f"\n### LoRAs: {len(models['loras'])} 个\n"
    for i, name in enumerate(models['loras'][:10], 1):
        info += f"{i}. {name}\n"
    if len(models['loras']) > 10:
        info += f"... 还有 {len(models['loras']) - 10} 个\n"
    
    info += f"\n### VAEs: {len(models['vaes'])} 个\n"
    for i, name in enumerate(models['vaes'][:10], 1):
        info += f"{i}. {name}\n"
    if len(models['vaes']) > 10:
        info += f"... 还有 {len(models['vaes']) - 10} 个\n"
    
    return info


def demo_generate(prompt, steps, cfg):
    """演示生成"""
    import time
    time.sleep(1)
    return f"""
✅ 演示完成！

**提示词:** {prompt}
**步数:** {steps}
**CFG:** {cfg}

这是演示界面，实际生成需要完整的 Genesis 引擎。
"""


# 创建界面
with gr.Blocks(title="Genesis Demo") as demo:
    gr.Markdown("# 🎨 Genesis AI 演示")
    
    with gr.Tab("生成"):
        prompt = gr.Textbox(label="提示词", value="a beautiful landscape")
        with gr.Row():
            steps = gr.Slider(1, 100, 20, label="步数")
            cfg = gr.Slider(1, 20, 7, label="CFG")
        btn = gr.Button("生成")
        output = gr.Markdown()
        btn.click(demo_generate, [prompt, steps, cfg], output)
    
    with gr.Tab("模型"):
        gr.Markdown(show_models())


if __name__ == "__main__":
    models = get_models()
    print(f"Checkpoints: {len(models['checkpoints'])}")
    print(f"LoRAs: {len(models['loras'])}")
    print(f"VAEs: {len(models['vaes'])}")
    print("\n启动界面...")
    
    demo.launch(
        server_port=7860,
        inbrowser=True
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 ComfyUI 模型的示例
演示如何在 Genesis 中使用 ComfyUI 的模型文件

前提条件:
1. 已配置 extra_model_paths.yaml
2. ComfyUI models 文件夹中有模型文件
"""

import sys
from pathlib import Path

# 添加 Genesis 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis import GenesisEngine, GenesisConfig
from genesis.core import folder_paths


def list_available_models():
    """列出所有可用的模型"""
    print("=" * 70)
    print("可用的模型列表")
    print("=" * 70)
    
    # Checkpoints
    checkpoints = folder_paths.get_filename_list('checkpoints')
    print(f"\n📦 Checkpoints ({len(checkpoints)}):")
    for i, name in enumerate(checkpoints[:10], 1):
        print(f"  {i}. {name}")
    if len(checkpoints) > 10:
        print(f"  ... 还有 {len(checkpoints) - 10} 个")
    
    # LoRAs
    loras = folder_paths.get_filename_list('loras')
    print(f"\n🎨 LoRAs ({len(loras)}):")
    for i, name in enumerate(loras[:10], 1):
        print(f"  {i}. {name}")
    if len(loras) > 10:
        print(f"  ... 还有 {len(loras) - 10} 个")
    
    # VAEs
    vaes = folder_paths.get_filename_list('vae')
    print(f"\n🖼️  VAEs ({len(vaes)}):")
    for i, name in enumerate(vaes[:10], 1):
        print(f"  {i}. {name}")
    if len(vaes) > 10:
        print(f"  ... 还有 {len(vaes) - 10} 个")
    
    print("\n" + "=" * 70)


def load_model_example():
    """加载模型的示例"""
    print("\n" + "=" * 70)
    print("模型加载示例")
    print("=" * 70)
    
    # 获取第一个可用的 checkpoint
    checkpoints = folder_paths.get_filename_list('checkpoints')
    
    if not checkpoints:
        print("\n⚠️  未找到任何 checkpoint 模型")
        print("   请在 ComfyUI models/checkpoints 文件夹中放置模型文件")
        return
    
    # 选择第一个模型
    model_name = checkpoints[0]
    model_path = folder_paths.get_full_path('checkpoints', model_name)
    
    print(f"\n选择的模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"文件存在: {'是' if Path(model_path).exists() else '否'}")
    
    # 创建 Genesis 引擎配置
    config = GenesisConfig(
        device='cuda',  # 或 'cpu'
        log_level='INFO'
    )
    
    print(f"\n创建 Genesis 引擎...")
    
    try:
        # 创建引擎
        engine = GenesisEngine(config)
        engine.initialize()
        
        print("✓ 引擎初始化成功")
        
        # 这里可以添加实际的模型加载和使用代码
        # 例如:
        # model = engine.load_checkpoint(model_path)
        # result = engine.generate(...)
        
        # 清理
        engine.cleanup()
        
        print("✓ 示例完成")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def dynamic_add_path_example():
    """动态添加路径的示例"""
    print("\n" + "=" * 70)
    print("动态添加模型路径示例")
    print("=" * 70)
    
    # 假设你有另一个模型文件夹
    custom_path = r"D:\MyModels\checkpoints"
    
    print(f"\n添加自定义路径: {custom_path}")
    
    # 动态添加路径
    folder_paths.add_model_folder_path('checkpoints', custom_path)
    
    # 查看更新后的路径
    all_paths = folder_paths.get_folder_paths('checkpoints')
    print(f"\n当前所有 checkpoints 路径:")
    for i, path in enumerate(all_paths, 1):
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {i}. [{exists}] {path}")
    
    print("\n提示: 这只是示例，实际使用时请修改为你的真实路径")


def main():
    """主函数"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║           Genesis - 使用 ComfyUI 模型示例                      ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    try:
        # 1. 列出可用模型
        list_available_models()
        
        # 2. 加载模型示例
        load_model_example()
        
        # 3. 动态添加路径示例
        # dynamic_add_path_example()
        
        print("\n" + "=" * 70)
        print("所有示例完成!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

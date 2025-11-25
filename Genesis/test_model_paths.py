#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型路径配置
验证 Genesis 是否能正确读取 ComfyUI 的模型文件夹

使用方法:
    python test_model_paths.py
"""

import sys
from pathlib import Path

# 添加 Genesis 到路径
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入 core 模块
from core import folder_paths


def test_model_paths():
    """测试模型路径配置"""
    print("=" * 70)
    print("Genesis 模型路径配置测试")
    print("=" * 70)
    
    # 测试的模型类型
    model_types = [
        'checkpoints',
        'loras',
        'vae',
        'clip',
        'text_encoders',
        'controlnet',
        'upscale_models',
        'embeddings',
    ]
    
    print("\n1. 检查配置的路径:")
    print("-" * 70)
    
    for model_type in model_types:
        paths = folder_paths.get_folder_paths(model_type)
        print(f"\n{model_type}:")
        if paths:
            for i, path in enumerate(paths, 1):
                exists = "✓" if Path(path).exists() else "✗"
                print(f"  {i}. [{exists}] {path}")
        else:
            print(f"  (未配置)")
    
    print("\n\n2. 检查可用的模型文件:")
    print("-" * 70)
    
    for model_type in model_types:
        files = folder_paths.get_filename_list(model_type)
        print(f"\n{model_type}: {len(files)} 个文件")
        if files:
            # 只显示前 5 个文件
            for i, filename in enumerate(files[:5], 1):
                print(f"  {i}. {filename}")
            if len(files) > 5:
                print(f"  ... 还有 {len(files) - 5} 个文件")
        else:
            print(f"  (未找到文件)")
    
    print("\n\n3. 测试获取完整路径:")
    print("-" * 70)
    
    # 尝试获取第一个 checkpoint 的完整路径
    checkpoints = folder_paths.get_filename_list('checkpoints')
    if checkpoints:
        first_checkpoint = checkpoints[0]
        full_path = folder_paths.get_full_path('checkpoints', first_checkpoint)
        print(f"\n示例: {first_checkpoint}")
        print(f"完整路径: {full_path}")
        print(f"文件存在: {'是' if Path(full_path).exists() else '否'}")
    else:
        print("\n未找到 checkpoint 文件")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    
    # 统计信息
    total_files = sum(len(folder_paths.get_filename_list(mt)) for mt in model_types)
    print(f"\n总计: {total_files} 个模型文件可用")
    
    if total_files == 0:
        print("\n⚠️  提示: 未找到任何模型文件")
        print("   请检查 extra_model_paths.yaml 配置是否正确")
        print("   或在 ComfyUI models 文件夹中放置模型文件")
    else:
        print("\n✓ 配置成功! Genesis 可以访问你的模型文件")


def show_directory_structure():
    """显示完整的目录结构"""
    print("\n\n4. 完整目录结构:")
    print("-" * 70)
    
    structure = folder_paths.get_directory_structure()
    for name, paths in sorted(structure.items()):
        print(f"\n{name}:")
        for path in paths:
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  [{exists}] {path}")


if __name__ == "__main__":
    try:
        test_model_paths()
        
        # 可选: 显示完整目录结构
        import sys
        if '--full' in sys.argv:
            show_directory_structure()
        else:
            print("\n提示: 使用 --full 参数查看完整目录结构")
            print("      python test_model_paths.py --full")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

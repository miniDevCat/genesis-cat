#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的模型路径测试脚本
直接测试 folder_paths 模块
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 直接导入 folder_paths 模块（避免导入整个 core 包）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "folder_paths", 
    current_dir / "core" / "folder_paths.py"
)
folder_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folder_paths)


def test_model_paths():
    """测试模型路径配置"""
    print("=" * 70)
    print("Genesis Model Paths Configuration Test")
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
    
    print("\n1. Configured Paths:")
    print("-" * 70)
    
    for model_type in model_types:
        paths = folder_paths.get_folder_paths(model_type)
        print(f"\n{model_type}:")
        if paths:
            for i, path in enumerate(paths, 1):
                exists = "OK" if Path(path).exists() else "NOT FOUND"
                print(f"  {i}. [{exists}] {path}")
        else:
            print(f"  (Not configured)")
    
    print("\n\n2. Available Model Files:")
    print("-" * 70)
    
    total_files = 0
    for model_type in model_types:
        files = folder_paths.get_filename_list(model_type)
        total_files += len(files)
        print(f"\n{model_type}: {len(files)} files")
        if files:
            # Show first 5 files
            for i, filename in enumerate(files[:5], 1):
                print(f"  {i}. {filename}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
        else:
            print(f"  (No files found)")
    
    print("\n\n3. Test Get Full Path:")
    print("-" * 70)
    
    # Try to get full path of first checkpoint
    checkpoints = folder_paths.get_filename_list('checkpoints')
    if checkpoints:
        first_checkpoint = checkpoints[0]
        full_path = folder_paths.get_full_path('checkpoints', first_checkpoint)
        print(f"\nExample: {first_checkpoint}")
        print(f"Full path: {full_path}")
        print(f"File exists: {'Yes' if Path(full_path).exists() else 'No'}")
    else:
        print("\nNo checkpoint files found")
    
    print("\n" + "=" * 70)
    print("Test Completed!")
    print("=" * 70)
    
    # Summary
    print(f"\nTotal: {total_files} model files available")
    
    if total_files == 0:
        print("\nWARNING: No model files found")
        print("  Please check extra_model_paths.yaml configuration")
        print("  or place model files in ComfyUI models folder")
    else:
        print("\nSUCCESS! Genesis can access your model files")
        print(f"  ComfyUI path: e:\\Comfyu3.13---test\\ComfyUI\\models")


if __name__ == "__main__":
    try:
        test_model_paths()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

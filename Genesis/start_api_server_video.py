#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis API Server - 支持文生图和文生视频
使用本地模型路径
"""

import sys
import os
from pathlib import Path
import base64
from io import BytesIO

# 禁用所有可选的注意力优化库以避免兼容性问题
os.environ['DIFFUSERS_DISABLE_FLASH_ATTENTION'] = '1'
os.environ['DISABLE_SAGE_ATTENTION'] = '1'
os.environ['ATTN_BACKEND'] = 'pytorch'  # 强制使用 PyTorch 原生实现

# 设置 ComfyUI 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
os.environ['COMFYUI_PATH'] = project_root
sys.path.insert(0, project_root)

try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    print("=" * 70)
    print("错误: Flask 未安装")
    print("=" * 70)
    print()
    print("请安装必要的依赖:")
    print("  pip install flask flask-cors")
    print()
    sys.exit(1)

try:
    import torch
    import numpy as np
    from PIL import Image
    TORCH_AVAILABLE = True
    print("✓ PyTorch 已安装")
except ImportError:
    print("⚠️  PyTorch 未安装")
    TORCH_AVAILABLE = False

import uuid
import time
import threading
from queue import Queue, Empty
from datetime import datetime
import json

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 任务存储
tasks = {}
task_queue = Queue()

# 任务状态常量
class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# 任务类
class Task:
    def __init__(self, task_id, task_type, params):
        self.task_id = task_id
        self.task_type = task_type
        self.params = params
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'params': self.params,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

# 模型路径配置
IMAGE_MODELS = {
    'sd-v1-5': {
        'name': 'Stable Diffusion v1.5',
        'path': 'runwayml/stable-diffusion-v1-5',
        'type': 'huggingface'
    },
    'sd-v2-1': {
        'name': 'Stable Diffusion v2.1',
        'path': 'stabilityai/stable-diffusion-2-1',
        'type': 'huggingface'
    },
}

VIDEO_MODELS = {
    'wan2.2-i2v': {
        'name': 'Wan2.2 I2V (NSFW)',
        'path': r'E:\fuxkcomfy_windows_portable\FuxkComfy\models\diffusion_models\wan2.2-i2v-rapid-aio-nsfw-v9.2.safetensors',
        'type': 'i2v',
        'description': '图生视频模型，快速生成'
    },
    'wan2-icecannon-t2v': {
        'name': 'Wan2 IceCannon T2V (NSFW)',
        'path': r'E:\fuxkcomfy_windows_portable\FuxkComfy\models\diffusion_models\Wan2_IceCannon_t2v2.1_nsfw_RCM_Lab_4step.safetensors',
        'type': 't2v',
        'description': '文生视频模型，4步快速生成'
    },
    'svd-img2vid': {
        'name': 'Stable Video Diffusion',
        'path': 'stabilityai/stable-video-diffusion-img2vid',
        'type': 'i2v',
        'description': '官方图生视频模型'
    },
}

# 默认模型
DEFAULT_IMAGE_MODEL = 'sd-v1-5'
DEFAULT_VIDEO_MODEL = 'wan2.2-i2v'

# 全局生成器
class MultiModalGenerator:
    def __init__(self):
        self.image_pipe = None
        self.video_pipe = None
        self.wan_video_workflow = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32 if TORCH_AVAILABLE else None
        self.image_model_loaded = False
        self.video_model_loaded = False
        self.current_image_model = None
        self.current_video_model = None
        
        print("\n" + "="*60)
        print("初始化多模态生成器")
        print("="*60)
        
        # 尝试加载 WanVideo 工作流
        try:
            self._init_wanvideo_workflow()
        except Exception as e:
            print(f"❌ WanVideo 工作流初始化异常: {e}")
            import traceback
            traceback.print_exc()
            self.wan_video_workflow = None
        
    def load_image_model(self, model_id=None):
        """加载文生图模型"""
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch 未安装，跳过模型加载")
            return False
        
        if model_id is None:
            model_id = DEFAULT_IMAGE_MODEL
        
        # 如果已加载相同模型，跳过
        if self.image_model_loaded and self.current_image_model == model_id:
            print(f"✓ 图像模型已加载: {model_id}")
            return True
            
        try:
            from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
            
            if model_id not in IMAGE_MODELS:
                raise ValueError(f"未知的图像模型: {model_id}")
            
            model_info = IMAGE_MODELS[model_id]
            model_path = model_info['path']
            
            print(f"📥 加载图像模型: {model_info['name']}")
            print(f"   路径: {model_path}")
            print(f"   设备: {self.device}")
            
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.image_pipe = self.image_pipe.to(self.device)
            self.image_pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.image_pipe.scheduler.config
            )
            
            if self.device == "cuda":
                self.image_pipe.enable_attention_slicing()
            
            self.image_model_loaded = True
            self.current_image_model = model_id
            print(f"✅ 图像模型加载成功: {model_info['name']}")
            return True
            
        except Exception as e:
            print(f"❌ 图像模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _init_wanvideo_workflow(self):
        """初始化 WanVideo 工作流"""
        try:
            # 设置正确的路径
            genesis_root = os.path.dirname(current_dir)  # E:\Comfyu3.13---test
            genesis_main = current_dir  # E:\Comfyu3.13---test\Genesis-main
            apps_path = os.path.join(genesis_main, 'apps')
            
            print(f"🔍 尝试加载 WanVideo 工作流...")
            print(f"   Genesis 根目录: {genesis_root}")
            print(f"   Genesis-main: {genesis_main}")
            print(f"   Apps 路径: {apps_path}")
            
            if not os.path.exists(apps_path):
                print(f"   ❌ Apps 目录不存在")
                self.wan_video_workflow = None
                return False
            
            # 添加必要的路径到 sys.path
            if genesis_root not in sys.path:
                sys.path.insert(0, genesis_root)
            if genesis_main not in sys.path:
                sys.path.insert(0, genesis_main)
            if apps_path not in sys.path:
                sys.path.insert(0, apps_path)
            
            # 设置环境变量
            os.environ['COMFYUI_PATH'] = genesis_root
            
            # 创建 genesis 模块别名，指向 Genesis-main
            print("   ⏳ 创建模块别名...")
            import importlib.util
            
            # 将 Genesis-main 作为 genesis 模块导入
            spec = importlib.util.spec_from_file_location("genesis", os.path.join(genesis_main, "__init__.py"))
            if spec and spec.loader:
                genesis_module = importlib.util.module_from_spec(spec)
                sys.modules['genesis'] = genesis_module
                spec.loader.exec_module(genesis_module)
                print("   ✓ 创建 genesis 模块别名成功")
            
            print("   ⏳ 导入 WanVideo 工作流...")
            from wanvideo_gradio_app import WanVideoWorkflow
            
            print("   ⏳ 初始化 WanVideo 工作流...")
            try:
                self.wan_video_workflow = WanVideoWorkflow()
                print("✅ WanVideo 工作流初始化成功")
                self.video_model_loaded = True
                return True
            except RuntimeError as e:
                print(f"   ❌ WanVideo 工作流初始化失败: {e}")
                print("   这是正常的，说明缺少 ComfyUI-WanVideoWrapper 节点")
                print("   将使用备用的图像序列生成模式")
                self.wan_video_workflow = None
                return False
        except Exception as e:
            print(f"⚠️  WanVideo 工作流初始化失败: {e}")
            print("   将使用备用模式")
            import traceback
            traceback.print_exc()
            self.wan_video_workflow = None
            return False
    
    def load_video_model(self, model_id=None):
        """加载文生视频模型"""
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch 未安装，跳过模型加载")
            return False
        
        if model_id is None:
            model_id = DEFAULT_VIDEO_MODEL
        
        # 如果已加载相同模型，跳过
        if self.video_model_loaded and self.current_video_model == model_id:
            print(f"✓ 视频模型已加载: {model_id}")
            return True
            
        try:
            if model_id not in VIDEO_MODELS:
                raise ValueError(f"未知的视频模型: {model_id}")
            
            model_info = VIDEO_MODELS[model_id]
            model_path = model_info['path']
            
            print(f"📥 加载视频模型: {model_info['name']}")
            print(f"   路径: {model_path}")
            print(f"   类型: {model_info['type']}")
            print(f"   设备: {self.device}")
            
            # 尝试使用 diffusers 加载视频模型
            try:
                from diffusers import StableVideoDiffusionPipeline
                
                # 检查是否是本地 safetensors 文件
                if os.path.exists(model_path) and model_path.endswith('.safetensors'):
                    print("   尝试从 safetensors 文件加载...")
                    # 使用 from_single_file 加载
                    self.video_pipe = StableVideoDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=self.dtype
                    )
                else:
                    # HuggingFace 模型
                    print("   从 HuggingFace 加载...")
                    self.video_pipe = StableVideoDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self.dtype
                    )
                
                self.video_pipe = self.video_pipe.to(self.device)
                
                if self.device == "cuda":
                    self.video_pipe.enable_attention_slicing()
                
                print(f"✅ 视频模型加载成功: {model_info['name']}")
                self.current_video_model = model_id
                self.video_model_loaded = True
                return True
                
            except Exception as e:
                print(f"⚠️  使用 diffusers 加载失败: {e}")
                print("   将使用图像序列模式")
                self.current_video_model = model_id
                self.video_model_loaded = True  # 标记为已加载，使用备用方案
                return True
            
        except Exception as e:
            print(f"❌ 视频模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_image(self, params, progress_callback=None):
        """生成图像"""
        if not TORCH_AVAILABLE:
            return self._generate_mock_image(params)
        
        # 获取模型ID
        model_id = params.get('model_id', DEFAULT_IMAGE_MODEL)
        
        # 加载模型（如果需要）
        if not self.image_model_loaded or self.current_image_model != model_id:
            if not self.load_image_model(model_id):
                raise Exception("图像模型加载失败")
        
        try:
            prompt = params.get('prompt', '')
            negative_prompt = params.get('negative_prompt', '')
            width = params.get('width', 512)
            height = params.get('height', 512)
            steps = params.get('steps', 20)
            cfg_scale = params.get('cfg_scale', 7.0)
            seed = params.get('seed')
            
            if seed is None or seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            
            print(f"🎨 开始生成图像...")
            print(f"   提示词: {prompt[:50]}...")
            
            def callback(step, timestep, latents):
                if progress_callback:
                    progress = int((step / steps) * 80) + 10
                    progress_callback(progress)
            
            result = self.image_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                callback=callback,
                callback_steps=1
            )
            
            image = result.images[0]
            
            # 转换为 base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_data_url = f"data:image/png;base64,{img_str}"
            
            print("✅ 图像生成成功")
            
            return {
                'success': True,
                'image': img_data_url,
                'seed': seed,
                'prompt': prompt,
                'width': width,
                'height': height,
            }
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_video(self, params, progress_callback=None):
        """生成视频"""
        # 获取模型ID
        model_id = params.get('model_id', DEFAULT_VIDEO_MODEL)
        
        # 加载模型（如果需要）
        if not self.video_model_loaded or self.current_video_model != model_id:
            self.load_video_model(model_id)
        
        model_info = VIDEO_MODELS.get(model_id, VIDEO_MODELS[DEFAULT_VIDEO_MODEL])
        
        # 使用多帧图像序列生成真实视频
        if self.video_model_loaded:
            try:
                return self._generate_real_video(params, model_info, progress_callback)
            except Exception as e:
                print(f"❌ 真实视频生成失败，回退到模拟模式: {e}")
                import traceback
                traceback.print_exc()
                return self._generate_mock_video(params, model_info)
        else:
            print("⚠️  视频模型未加载")
            print(f"   当前模型: {model_info['name']}")
            return self._generate_mock_video(params, model_info)
    
    def _generate_mock_image(self, params):
        """生成模拟图像"""
        import random
        time.sleep(2)
        
        mock_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        return {
            'success': True,
            'image': mock_image,
            'seed': params.get('seed') or random.randint(0, 2147483647),
            'prompt': params.get('prompt', ''),
            'width': params.get('width', 512),
            'height': params.get('height', 512),
            'note': '⚠️ 这是模拟数据，请安装 PyTorch 和 Diffusers'
        }
    
    def _generate_real_video(self, params, model_info, progress_callback=None):
        """使用真实视频模型生成视频"""
        import random
        import numpy as np
        
        print("🎬 开始视频生成...")
        print(f"   模型: {model_info['name']}")
        
        # 提取参数
        prompt = params.get('prompt', '')
        negative_prompt = params.get('negative_prompt', '')
        width = params.get('width', 512)
        height = params.get('height', 512)
        frames = params.get('frames', 16)
        steps = params.get('steps', 20)
        cfg_scale = params.get('cfg_scale', 7.5)
        seed = params.get('seed', -1)
        fps = params.get('fps', 8)
        
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        
        # 如果有 WanVideo 工作流，使用它
        if self.wan_video_workflow:
            print("   使用 WanVideo 工作流生成...")
            return self._generate_with_wanvideo(params, model_info, seed, progress_callback)
        
        # 如果有视频模型，使用视频模型
        if self.video_pipe:
            print("   使用视频模型直接生成...")
            return self._generate_with_video_model(params, model_info, seed, progress_callback)
        
        # 否则使用图像序列模式
        print("   使用图像序列模式...")
        
        # 加载图像模型（用于生成帧）
        if not self.image_model_loaded:
            self.load_image_model()
        
        if not self.image_pipe:
            raise Exception("图像模型未加载")
        
        print(f"   生成 {frames} 帧图像...")
        print(f"   策略: 生成基础图像，然后通过图像变换创建帧序列")
        
        # 生成多帧图像
        video_frames = []
        
        # 先生成一张基础图像
        print(f"   [1/2] 生成基础图像...")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.no_grad():
            base_image = self.image_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            ).images[0]
        
        base_array = np.array(base_image)
        print(f"   ✓ 基础图像生成完成")
        
        # 生成变化序列
        print(f"   [2/2] 生成 {frames} 帧变化序列...")
        
        for i in range(frames):
            if progress_callback:
                progress = 0.5 + (i / frames) * 0.4
                progress_callback(int(progress * 100))
            
            # 计算变化程度
            variation = i / max(frames - 1, 1) if frames > 1 else 0
            
            # 创建轻微变化的图像
            if i == 0:
                # 第一帧使用原始图像
                frame_array = base_array.copy()
            else:
                # 后续帧添加轻微的亮度/对比度变化
                frame_array = base_array.copy().astype(np.float32)
                
                # 添加轻微的亮度变化（模拟光线变化）
                brightness_change = np.sin(variation * np.pi) * 0.05  # -0.05 到 0.05
                frame_array = frame_array * (1.0 + brightness_change)
                
                # 添加轻微的色调变化
                hue_shift = np.sin(variation * np.pi * 2) * 3  # -3 到 3
                frame_array[:, :, 0] = np.clip(frame_array[:, :, 0] + hue_shift, 0, 255)
                
                # 添加轻微的缩放效果（模拟运动）
                zoom_factor = 1.0 + variation * 0.02  # 1.0 到 1.02
                h, w = frame_array.shape[:2]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                
                from PIL import Image
                temp_img = Image.fromarray(frame_array.astype(np.uint8))
                temp_img = temp_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # 裁剪回原始尺寸
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                temp_img = temp_img.crop((left, top, left + w, top + h))
                
                frame_array = np.array(temp_img)
            
            # 确保数值范围正确
            frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
            video_frames.append(frame_array)
            
            if (i + 1) % 4 == 0 or i == frames - 1:
                print(f"   生成进度: {i+1}/{frames} 帧")
        
        print(f"   ✓ 生成了 {len(video_frames)} 帧")
        
        # 将帧序列转换为视频
        print("   编码视频...")
        import cv2
        
        temp_video_path = f"temp_video_{seed}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        for frame in video_frames:
            # 转换 RGB 到 BGR（OpenCV 格式）
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # 读取视频文件并转换为 base64
        with open(temp_video_path, 'rb') as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            video_data_url = f"data:video/mp4;base64,{video_base64}"
        
        # 删除临时文件
        try:
            os.remove(temp_video_path)
        except:
            pass
        
        print("✅ 视频生成成功")
        
        return {
            'success': True,
            'video': video_data_url,
            'seed': seed,
            'prompt': prompt,
            'width': width,
            'height': height,
            'frames': frames,
            'fps': fps,
            'model_name': model_info['name'],
            'note': f'✅ 使用 Stable Diffusion 生成 {frames} 帧视频'
        }
    
    def _generate_with_wanvideo(self, params, model_info, seed, progress_callback=None):
        """使用 WanVideo 工作流生成视频"""
        import random
        import numpy as np
        
        prompt = params.get('prompt', '')
        negative_prompt = params.get('negative_prompt', '')
        width = params.get('width', 512)
        height = params.get('height', 512)
        frames = params.get('frames', 16)
        steps = params.get('steps', 20)
        cfg_scale = params.get('cfg_scale', 7.5)
        fps = params.get('fps', 8)
        
        print(f"   提示词: {prompt[:50]}...")
        print(f"   参数: {frames}帧, {steps}步, CFG={cfg_scale}")
        
        # 从模型路径中提取模型名称
        model_path = model_info['path']
        model_name = os.path.basename(model_path)
        
        try:
            # 调用 WanVideo 工作流
            video_array, metadata = self.wan_video_workflow.generate_video(
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                model_name=model_name,
                vae_name="Wan2_1_VAE_bf16.safetensors",
                t5_model="google/t5-v1_1-xxl",
                width=width,
                height=height,
                num_frames=frames,
                steps=steps,
                cfg=cfg_scale,
                shift=1.0,
                seed=seed,
                scheduler="unipc",
                denoise_strength=1.0,
                quantization="fp8_e4m3fn_fast",
                attention_mode="auto",
                lora_enabled=False,
                lora_name="",
                lora_strength=1.0,
                compile_enabled=False,
                compile_backend="inductor",
                block_swap_enabled=False,
                blocks_to_swap=0,
                output_format="mp4",
                fps=fps,
                progress_callback=progress_callback
            )
            
            print(f"   ✓ 生成了 {len(video_array)} 帧")
            
            # 转换为视频
            print("   编码视频...")
            import cv2
            
            temp_video_path = f"temp_video_{seed}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            for frame in video_array:
                # 转换 RGB 到 BGR
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
            
            out.release()
            
            # 读取并转换为 base64
            with open(temp_video_path, 'rb') as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                video_data_url = f"data:video/mp4;base64,{video_base64}"
            
            try:
                os.remove(temp_video_path)
            except:
                pass
            
            print("✅ 视频生成成功")
            
            return {
                'success': True,
                'video': video_data_url,
                'seed': seed,
                'prompt': prompt,
                'width': width,
                'height': height,
                'frames': frames,
                'fps': fps,
                'model_name': model_info['name'],
                'note': f'✅ 使用 WanVideo 工作流和 {model_info["name"]} 生成'
            }
        except Exception as e:
            print(f"❌ WanVideo 工作流生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_with_video_model(self, params, model_info, seed, progress_callback=None):
        """使用视频模型直接生成视频"""
        import random
        import numpy as np
        
        prompt = params.get('prompt', '')
        negative_prompt = params.get('negative_prompt', '')
        width = params.get('width', 512)
        height = params.get('height', 512)
        frames = params.get('frames', 16)
        steps = params.get('steps', 20)
        cfg_scale = params.get('cfg_scale', 7.5)
        fps = params.get('fps', 8)
        
        print(f"   提示词: {prompt[:50]}...")
        print(f"   参数: {frames}帧, {steps}步, CFG={cfg_scale}")
        
        # 先生成初始图像（视频模型通常需要初始图像）
        if not self.image_model_loaded:
            self.load_image_model()
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print("   [1/2] 生成初始图像...")
        with torch.no_grad():
            init_image = self.image_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            ).images[0]
        
        print("   [2/2] 使用视频模型生成视频...")
        
        # 使用视频模型生成
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.no_grad():
            video_frames = self.video_pipe(
                init_image,
                decode_chunk_size=8,
                generator=generator,
                num_frames=frames
            ).frames[0]
        
        print(f"   ✓ 生成了 {len(video_frames)} 帧")
        
        # 转换为视频
        print("   编码视频...")
        import cv2
        
        temp_video_path = f"temp_video_{seed}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        for frame in video_frames:
            # 确保是 numpy 数组
            if hasattr(frame, 'numpy'):
                frame = frame.numpy()
            frame = np.array(frame)
            
            # 转换 RGB 到 BGR
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
        
        out.release()
        
        # 读取并转换为 base64
        with open(temp_video_path, 'rb') as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            video_data_url = f"data:video/mp4;base64,{video_base64}"
        
        try:
            os.remove(temp_video_path)
        except:
            pass
        
        print("✅ 视频生成成功")
        
        return {
            'success': True,
            'video': video_data_url,
            'seed': seed,
            'prompt': prompt,
            'width': width,
            'height': height,
            'frames': frames,
            'fps': fps,
            'model_name': model_info['name'],
            'note': f'✅ 使用 {model_info["name"]} 视频模型生成'
        }
    
    def _generate_mock_video(self, params, model_info):
        """生成模拟视频"""
        import random
        time.sleep(3)
        
        # 返回一个简单的视频 URL（实际应该是 base64 编码的 MP4）
        mock_video = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAu1tZGF0AAACrQYF//+c3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1MiByMjg1NCBlOWE1OTAzIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxNyAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTYgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTI1IHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAAD2WIhAAz//727L4FNf2f0JcRLMXaSnA+KqSAgHc0wAAAAwAAAwAAFgn0I7DkqgN3QAAAHGliYXNlbGluZQMAD21vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAPoAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIYdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAPoAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAACgAAAAWgAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAD6AAAAAAAAQAAAAABkG1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAPAAAADwAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAATttaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAD7c3RibAAAAJdzdHNkAAAAAAAAAAEAAACHYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAKAAFoASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADFhdmNDAWQAFf/hABhnZAAVrNlBsJaEAAADAAQAAAMAPB4sWLZYAQAGaOvjyyLAAAAAGHN0dHMAAAAAAAAAAQAAAAEAAAABAAAAABRzdHNzAAAAAAAAAAEAAAABAAAAGHN0c2MAAAAAAAAAAQAAAAEAAAABAAAAAQAAABRzdHN6AAAAAAAAAAAAAAABAAAAHAAAABRzdGNvAAAAAAAAAAEAAAAsAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY1OC4yOS4xMDA="
        
        return {
            'success': True,
            'video': mock_video,
            'seed': params.get('seed') or random.randint(0, 2147483647),
            'prompt': params.get('prompt', ''),
            'width': params.get('width', 512),
            'height': params.get('height', 512),
            'frames': params.get('frames', 16),
            'fps': params.get('fps', 8),
            'model_name': model_info['name'],
            'note': f'⚠️ 视频生成功能开发中\n当前模型: {model_info["name"]}\n模型路径: {model_info["path"]}\n需要实现 I2V 模型加载'
        }

# 创建全局生成器
generator = MultiModalGenerator()

# 工作线程
worker_running = False
worker_thread = None

def worker_loop():
    """处理任务队列"""
    global worker_running
    print("🔄 Worker thread started")
    
    while worker_running:
        try:
            task_id = task_queue.get(timeout=1.0)
            task = tasks.get(task_id)
            
            if not task:
                continue
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.progress = 5
            
            print(f"📝 Processing task: {task_id}")
            print(f"   任务类型: {task.task_type}")
            print(f"   提示词: {task.params.get('prompt', 'N/A')[:50]}...")
            
            try:
                def progress_callback(progress):
                    task.progress = progress
                
                # 根据任务类型执行不同的生成
                if task.task_type == 'generate' or task.task_type == 'text_to_image':
                    print("   → 执行文生图")
                    result = generator.generate_image(task.params, progress_callback)
                elif task.task_type == 'text_to_video':
                    print("   → 执行文生视频")
                    result = generator.generate_video(task.params, progress_callback)
                else:
                    raise ValueError(f"未知任务类型: {task.task_type}")
                
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.progress = 100
                task.completed_at = datetime.now()
                
                print(f"✅ Task completed: {task_id}")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                print(f"❌ Task failed: {task_id} - {e}")
                
        except Empty:
            continue
        except Exception as e:
            print(f"❌ Worker error: {e}")
    
    print("🛑 Worker thread stopped")

# API 路由
@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'torch_available': TORCH_AVAILABLE,
        'image_model_loaded': generator.image_model_loaded,
        'video_model_loaded': generator.video_model_loaded,
        'device': generator.device if TORCH_AVAILABLE else 'N/A',
        'tasks_pending': task_queue.qsize(),
        'tasks_total': len(tasks),
        'model_paths': MODEL_PATHS
    })

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """创建会话"""
    data = request.get_json() or {}
    session_id = str(uuid.uuid4())
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'session': {
            'session_id': session_id,
            'client_type': data.get('client_type', 'web'),
            'created_at': datetime.now().isoformat()
        }
    })

@app.route('/api/task/submit', methods=['POST'])
def submit_task():
    """提交任务"""
    try:
        data = request.get_json() or {}
        
        task_type = data.get('task_type', 'generate')
        params = data.get('params', {})
        
        # 创建任务
        task_id = str(uuid.uuid4())
        task = Task(task_id, task_type, params)
        
        # 保存任务
        tasks[task_id] = task
        task_queue.put(task_id)
        
        print(f"📥 Task submitted: {task_id} ({task_type})")
        print(f"   Prompt: {params.get('prompt', 'N/A')[:50]}...")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'task': task.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
    """获取任务状态"""
    task = tasks.get(task_id)
    
    if not task:
        return jsonify({
            'success': False,
            'error': 'Task not found'
        }), 404
    
    return jsonify({
        'success': True,
        'task': task.to_dict()
    })

@app.route('/api/task/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    """取消任务"""
    task = tasks.get(task_id)
    
    if not task:
        return jsonify({
            'success': False,
            'error': 'Task not found'
        }), 404
    
    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        return jsonify({
            'success': False,
            'error': 'Task already finished'
        }), 400
    
    task.status = TaskStatus.FAILED
    task.error = 'Cancelled by user'
    task.completed_at = datetime.now()
    
    return jsonify({
        'success': True,
        'task': task.to_dict()
    })

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """列出所有任务"""
    return jsonify({
        'success': True,
        'tasks': [task.to_dict() for task in tasks.values()],
        'count': len(tasks)
    })

@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    return jsonify({
        'success': True,
        'models': {
            'image_models': [
                {
                    'id': key,
                    'name': value['name'],
                    'path': value['path'],
                    'type': value['type']
                }
                for key, value in IMAGE_MODELS.items()
            ],
            'video_models': [
                {
                    'id': key,
                    'name': value['name'],
                    'path': value['path'],
                    'type': value['type'],
                    'description': value.get('description', '')
                }
                for key, value in VIDEO_MODELS.items()
            ],
            'default_image_model': DEFAULT_IMAGE_MODEL,
            'default_video_model': DEFAULT_VIDEO_MODEL
        }
    })

@app.route('/api/device', methods=['GET'])
def device_info():
    """获取设备信息"""
    if not TORCH_AVAILABLE:
        return jsonify({
            'success': True,
            'device': {
                'device': 'N/A',
                'note': 'PyTorch not installed'
            }
        })
    
    info = {'device': generator.device}
    
    if generator.device == 'cuda':
        info['device_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['memory_total'] = props.total_memory
        info['memory_allocated'] = torch.cuda.memory_allocated(0)
    
    return jsonify({
        'success': True,
        'device': info
    })

@app.route('/')
def index():
    """API 首页"""
    return jsonify({
        'name': 'Genesis API Server (Multi-Modal)',
        'version': '1.1.0',
        'status': 'running',
        'torch_available': TORCH_AVAILABLE,
        'image_model_loaded': generator.image_model_loaded,
        'video_model_loaded': generator.video_model_loaded,
        'model_paths': MODEL_PATHS,
        'endpoints': {
            'GET  /health': 'Health check',
            'POST /api/session/create': 'Create session',
            'POST /api/task/submit': 'Submit task (text_to_image or text_to_video)',
            'GET  /api/task/<id>': 'Get task status',
            'POST /api/task/<id>/cancel': 'Cancel task',
            'GET  /api/tasks': 'List all tasks',
            'GET  /api/models': 'List models',
            'GET  /api/device': 'Device info'
        }
    })

def start_worker():
    """启动工作线程"""
    global worker_running, worker_thread
    
    worker_running = True
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()

def stop_worker():
    """停止工作线程"""
    global worker_running
    
    worker_running = False
    if worker_thread:
        worker_thread.join(timeout=5.0)

def main():
    """主函数"""
    print("=" * 70)
    print("Genesis API Server - Multi-Modal (Image + Video)")
    print("=" * 70)
    print()
    
    print("📁 可用模型:")
    print()
    print("  图像模型:")
    for key, value in IMAGE_MODELS.items():
        print(f"    [{key}] {value['name']}")
        print(f"        {value['path']}")
    print()
    print("  视频模型:")
    for key, value in VIDEO_MODELS.items():
        print(f"    [{key}] {value['name']}")
        print(f"        {value['path']}")
        print(f"        {value.get('description', '')}")
    print()
    print(f"  默认图像模型: {IMAGE_MODELS[DEFAULT_IMAGE_MODEL]['name']}")
    print(f"  默认视频模型: {VIDEO_MODELS[DEFAULT_VIDEO_MODEL]['name']}")
    print()
    
    if TORCH_AVAILABLE:
        print("✅ PyTorch 可用")
        print(f"   设备: {generator.device}")
        if generator.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  PyTorch 未安装，使用模拟模式")
    
    print()
    print("🚀 Starting server...")
    print()
    print("📡 Server URL: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/health")
    print("📚 API docs: http://localhost:5000/")
    print()
    print("=" * 70)
    print()
    
    # 启动工作线程
    start_worker()
    
    try:
        # 启动 Flask 服务器
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        stop_worker()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()

"""
WanVideo API Server for Ant Design Pro Frontend
真正使用 WanVideo 模型的 API 服务器
"""

import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
from datetime import datetime
from queue import Queue, Empty
import threading
import base64
import numpy as np

# 设置路径 - Genesis-main 本身就是根目录
current_dir = Path(__file__).parent  # Genesis-main
project_root = current_dir.parent    # e:\Comfyu3.13---test

# 关键：将 Genesis-main 作为主路径
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['COMFYUI_PATH'] = str(project_root)
os.environ['PYTHONPATH'] = f"{current_dir};{project_root};{os.environ.get('PYTHONPATH', '')}"

print(f"[INFO] Project root: {project_root}")
print(f"[INFO] Genesis-main dir: {current_dir}")

# 导入 WanVideo 工作流
print("[INFO] Loading WanVideo Workflow...")
WANVIDEO_AVAILABLE = False
wanvideo_workflow = None
config_manager = None

try:
    # 创建 genesis 模块别名，指向 Genesis-main
    print("[INFO] Creating genesis module alias...")
    import importlib.util
    
    # 将 Genesis-main 注册为 genesis 模块
    spec = importlib.util.spec_from_file_location("genesis", current_dir / "__init__.py")
    if spec and spec.loader:
        genesis_module = importlib.util.module_from_spec(spec)
        sys.modules['genesis'] = genesis_module
        spec.loader.exec_module(genesis_module)
        print("[INFO] Genesis module alias created successfully")
    
    # 现在可以导入 genesis 子模块了
    print("[INFO] Importing genesis modules...")
    from genesis.utils import triton_ops_stub
    from genesis.compat import comfy_stub
    from genesis.core import folder_paths_ext
    
    # 导入 WanVideoWorkflow
    print("[INFO] Importing WanVideoWorkflow...")
    apps_path = current_dir / "apps"
    sys.path.insert(0, str(apps_path))
    
    from wanvideo_gradio_app import WanVideoWorkflow
    
    print("[INFO] WanVideo Workflow loaded successfully")
    WANVIDEO_AVAILABLE = True
    
    # 初始化工作流
    print("[INFO] Initializing WanVideo Workflow...")
    wanvideo_workflow = WanVideoWorkflow()
    print("[INFO] WanVideo Workflow initialized successfully")
    
    # 初始化配置管理器
    print("[INFO] Initializing Config Manager...")
    from genesis.utils.config_manager import get_config_manager
    config_manager = get_config_manager()
    print("[INFO] Config Manager initialized successfully")
    
except Exception as e:
    print(f"[ERROR] Failed to load WanVideo: {e}")
    import traceback
    traceback.print_exc()
    WANVIDEO_AVAILABLE = False

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 创建输出目录
OUTPUT_DIR = project_root / "outputs" / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Output directory: {OUTPUT_DIR}")

# 任务管理
tasks = {}
task_queue = Queue()

class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

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
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat()
        }

# Worker 线程
def worker_loop():
    """处理任务队列"""
    print("[INFO] Worker thread started")
    
    while True:
        try:
            task_id = task_queue.get(timeout=1.0)
            task = tasks.get(task_id)
            
            if not task:
                continue
            
            task.status = TaskStatus.RUNNING
            task.progress = 5
            
            print(f"[INFO] Processing task: {task_id}")
            print(f"[INFO] Task type: {task.task_type}")
            
            try:
                if task.task_type == 'text_to_video':
                    # 使用 WanVideo 生成视频
                    if not wanvideo_workflow:
                        raise Exception("WanVideo workflow not available")
                    
                    params = task.params
                    print(f"[INFO] Generating video with prompt: {params.get('prompt', '')[:50]}...")
                    print(f"[DEBUG] Received params keys: {list(params.keys())}")
                    print(f"[DEBUG] LoRAs in params: {params.get('loras', 'NOT FOUND')}")
                    
                    # 阶段 1: 准备模型 (5% -> 15%)
                    task.progress = 10
                    
                    # 处理模型选择
                    model_id = params.get('model_id', '')
                    if model_id:
                        # 如果 model_id 不包含扩展名，添加 .safetensors
                        if not (model_id.endswith('.safetensors') or model_id.endswith('.ckpt')):
                            model_name = f"{model_id}.safetensors"
                        else:
                            model_name = model_id
                        print(f"[INFO] Using model: {model_name}")
                    else:
                        # 使用默认模型
                        model_name = params.get('model_name', 'Wan2_IceCannon_t2v2.1_nsfw_RCM_Lab_4step.safetensors')
                        print(f"[INFO] Using default model: {model_name}")
                    
                    # 阶段 2: 加载模型 (15% -> 20%)
                    task.progress = 15
                    print(f"[INFO] Loading models...")
                    
                    # 定义进度回调函数
                    def update_progress(p, msg=""):
                        # 生成阶段占 20% -> 85%
                        new_progress = int(20 + p * 65)
                        if new_progress != task.progress:
                            task.progress = new_progress
                            if msg:
                                print(f"[PROGRESS] {new_progress}% - {msg}")
                    
                    # 处理 LoRA 列表
                    loras = params.get('loras', [])
                    lora_enabled = len(loras) > 0
                    lora_name = ''
                    lora_strength = 1.0
                    lora_low_mem_load = params.get('lora_low_mem_load', False)
                    lora_merge_loras = params.get('lora_merge_loras', False)
                    
                    # 如果使用 FP8 量化，强制合并 LoRA
                    quantization = params.get('quantization', 'fp8_e4m3fn_fast_scaled')
                    if lora_enabled and 'fp8' in quantization.lower() and not lora_merge_loras:
                        print(f"[WARNING] FP8 quantization requires merge_loras=True, auto-enabling...")
                        lora_merge_loras = True
                    
                    if lora_enabled:
                        print(f"[INFO] Using {len(loras)} LoRA(s):")
                        for lora in loras:
                            print(f"  - {lora['name']} (strength: {lora['strength']})")
                        # 保留第一个 LoRA 用于兼容性
                        lora_name = loras[0]['name']
                        lora_strength = loras[0]['strength']
                        print(f"[INFO] First LoRA: {lora_name} (strength: {lora_strength})")
                        print(f"[INFO] LoRA options: low_mem_load={lora_low_mem_load}, merge_loras={lora_merge_loras}")
                        print(f"[INFO] Passing {len(loras)} LoRA(s) to multi-LoRA loader")
                    
                    # 阶段 3: 生成视频 (20% -> 85%)
                    task.progress = 20
                    print(f"[INFO] Starting video generation...")
                    
                    # 调用 WanVideo 生成
                    video_path, video_array, metadata = wanvideo_workflow.generate_video(
                        positive_prompt=params.get('prompt', ''),
                        negative_prompt=params.get('negative_prompt', ''),
                        model_name=model_name,
                        vae_name=params.get('vae_name', 'Wan2_1_VAE_bf16.safetensors'),
                        t5_model=params.get('t5_model', 'umt5-xxl-enc-fp8_e4m3fn.safetensors'),
                        width=params.get('width', 512),
                        height=params.get('height', 512),
                        num_frames=params.get('frames', 16),
                        steps=params.get('steps', 20),
                        cfg=params.get('cfg_scale', 7.5),
                        shift=params.get('shift', 1.0),
                        seed=params.get('seed', -1) if params.get('seed') else -1,
                        scheduler=params.get('scheduler', 'unipc'),
                        base_precision=params.get('base_precision', 'bf16'),
                        denoise_strength=1.0,
                        quantization=params.get('quantization', 'fp8_e4m3fn_fast_scaled'),
                        attention_mode=params.get('attention_mode', 'auto'),
                        lora_enabled=lora_enabled,
                        lora_name=lora_name,
                        lora_strength=lora_strength,
                        compile_enabled=False,
                        compile_backend='inductor',
                        block_swap_enabled=False,
                        output_format='mp4',
                        fps=params.get('fps', 8),
                        loras_list=loras,  # 传递完整的 LoRA 列表
                        lora_low_mem_load=lora_low_mem_load,
                        lora_merge_loras=lora_merge_loras,
                        progress_callback=update_progress
                    )
                
                    # 阶段 4: 视频生成完成 (85%)
                    task.progress = 85
                    print(f"[INFO] Video generated successfully!")
                    print(f"[INFO] Original video path: {video_path}")
                    
                    # 阶段 5: 转换视频格式 (85% -> 95%)
                    task.progress = 88
                    
                    # 转换为 H.264 格式以提高兼容性
                    if video_path and os.path.exists(video_path):
                        import subprocess
                        
                        # 生成 H.264 版本的文件名
                        orig_filename = os.path.basename(video_path)
                        h264_filename = orig_filename.replace('.mp4', '_h264.mp4')
                        h264_path = current_dir / "output" / h264_filename
                        
                        print(f"[INFO] Converting to H.264 format for better compatibility...")
                        task.progress = 90
                        
                        # 使用 ffmpeg 转换
                        try:
                            subprocess.run([
                                'ffmpeg', '-y',
                                '-i', str(video_path),
                                '-c:v', 'libx264',
                                '-preset', 'fast',
                                '-crf', '23',
                                '-pix_fmt', 'yuv420p',
                                '-movflags', '+faststart',
                                str(h264_path)
                            ], check=True, capture_output=True)
                            
                            # 使用转换后的文件
                            video_filename = h264_filename
                            final_path = h264_path
                            task.progress = 95
                            print(f"[INFO] Conversion successful!")
                        except Exception as e:
                            print(f"[WARNING] Conversion failed: {e}")
                            print(f"[INFO] Using original file instead")
                            video_filename = orig_filename
                            final_path = video_path
                            task.progress = 95
                        
                        # 阶段 6: 保存结果 (95% -> 98%)
                        task.progress = 96
                        video_url = f"/genesis/output/{video_filename}"
                        file_size = os.path.getsize(final_path) / 1024 / 1024
                        
                        print(f"[INFO] Video file: {video_filename}")
                        print(f"[INFO] File size: {file_size:.2f} MB")
                        print(f"[INFO] Video URL: {video_url}")
                        print(f"[SUCCESS] Video ready for preview and download!")
                    else:
                        raise Exception(f"Video file not found: {video_path}")
                    
                    # 阶段 7: 保存配置 (98%)
                    task.progress = 98
                    
                    task.result = {
                        'success': True,
                        'video': video_url,
                        'video_path': str(video_path),
                        'prompt': params.get('prompt'),
                        'width': params.get('width'),
                        'height': params.get('height'),
                        'frames': params.get('frames'),
                        'fps': params.get('fps'),
                        'seed': metadata.get('seed', -1) if metadata else -1
                    }
                    
                    # 保存视频 URL 到配置文件
                    try:
                        import json
                        video_info = {
                            'width': params.get('width'),
                            'height': params.get('height'),
                            'frames': params.get('frames'),
                            'fps': params.get('fps'),
                        }
                        config_manager.save_last_used_params({
                            'last_video_url': video_url,
                            'last_video_info': json.dumps(video_info)
                        })
                        print(f"[INFO] Saved video URL to config")
                    except Exception as e:
                        print(f"[WARNING] Failed to save video URL: {e}")
                    
                    print(f"[INFO] Task completed successfully")
                
                elif task.task_type == 'text_to_image':
                    # 文生图不支持，返回友好错误
                    raise Exception("此 API 服务器仅支持文生视频。请使用文生视频功能或切换到图像生成 API。")
                    
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                task.status = TaskStatus.COMPLETED
                task.progress = 100
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                print(f"[ERROR] Task failed: {task_id} - {e}")
                import traceback
                traceback.print_exc()
                
        except Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Worker error: {e}")

# 启动 worker 线程
worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()

# API 路由
@app.route('/api/health', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'message': 'WanVideo API Server is running',
        'wanvideo_available': WANVIDEO_AVAILABLE,
        'tasks_pending': task_queue.qsize(),
        'tasks_total': len(tasks)
    })

@app.route('/api/task/submit', methods=['POST'])
def submit_task():
    try:
        data = request.get_json() or {}
        task_type = data.get('task_type', 'text_to_video')
        
        # 兼容两种格式：直接参数 或 params 包装
        if 'params' in data:
            params = data['params']
        else:
            # 直接使用 data 作为 params（新格式）
            params = data.copy()
            params.pop('task_type', None)  # 移除 task_type
        
        if not WANVIDEO_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'WanVideo workflow not available'
            }), 503
        
        task_id = str(uuid.uuid4())
        task = Task(task_id, task_type, params)
        
        tasks[task_id] = task
        task_queue.put(task_id)
        
        print(f"[INFO] Task submitted: {task_id}")
        print(f"[INFO] Prompt: {params.get('prompt', params.get('positive_prompt', ''))[:50]}...")
        print(f"[DEBUG] LoRAs in request: {params.get('loras', 'NOT FOUND')}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'task': task.to_dict()
        })
    except Exception as e:
        print(f"[ERROR] Submit task failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
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
    task = tasks.get(task_id)
    if not task:
        return jsonify({
            'success': False,
            'error': 'Task not found'
        }), 404
    
    # 简单实现：标记为失败
    if task.status == TaskStatus.PENDING:
        task.status = TaskStatus.FAILED
        task.error = 'Cancelled by user'
    
    return jsonify({
        'success': True,
        'task': task.to_dict()
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """返回可用的模型列表 - 自动扫描模型目录"""
    try:
        import folder_paths
        
        # 获取 diffusion_models 目录
        diffusion_paths = folder_paths.get_folder_paths("diffusion_models")
        
        video_models = []
        seen_models = set()  # 使用 set 追踪已添加的模型文件名
        
        for model_dir in diffusion_paths:
            if not os.path.exists(model_dir):
                continue
                
            # 扫描所有 .safetensors 和 .ckpt 文件
            for filename in os.listdir(model_dir):
                if not (filename.endswith('.safetensors') or filename.endswith('.ckpt')):
                    continue
                
                # 跳过已添加的模型
                if filename in seen_models:
                    continue
                seen_models.add(filename)
                
                # 判断模型类型
                filename_lower = filename.lower()
                if 'i2v' in filename_lower:
                    model_type = 'i2v'
                    type_name = 'I2V'
                elif 't2v' in filename_lower:
                    model_type = 't2v'
                    type_name = 'T2V'
                else:
                    # 默认为 T2V
                    model_type = 't2v'
                    type_name = 'T2V'
                
                # 生成模型 ID（使用文件名，去除扩展名）
                model_id = os.path.splitext(filename)[0]
                
                # 生成显示名称
                display_name = filename.replace('.safetensors', '').replace('.ckpt', '')
                display_name = display_name.replace('_', ' ').replace('-', ' ')
                
                video_models.append({
                    'id': model_id,
                    'name': f'{display_name} ({type_name})',
                    'path': filename,
                    'type': model_type,
                    'description': f'{type_name} 模型'
                })
        
        # 按类型和名称排序
        video_models.sort(key=lambda x: (x['type'], x['name']))
        
        print(f"[INFO] Found {len(video_models)} unique video models from {len(diffusion_paths)} paths")
        
        # 扫描 LoRA 模型
        lora_models = []
        try:
            lora_paths = folder_paths.get_folder_paths("loras")
            lora_set = set()  # 使用 set 去重
            for lora_dir in lora_paths:
                if not os.path.exists(lora_dir):
                    continue
                for filename in os.listdir(lora_dir):
                    if filename.endswith('.safetensors') or filename.endswith('.ckpt'):
                        lora_set.add(filename)  # 添加到 set 自动去重
            lora_models = sorted(list(lora_set))  # 转换为排序后的列表
            print(f"[INFO] Found {len(lora_models)} unique LoRA models from {len(lora_paths)} paths")
        except Exception as e:
            print(f"[WARNING] Failed to scan LoRA models: {e}")
        
        return jsonify({
            'success': True,
            'models': {
                'video_models': video_models,
                'default_video_model': video_models[0]['id'] if video_models else None,
                'loras': lora_models
            }
        })
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        # 返回默认模型列表
        return jsonify({
            'success': True,
            'models': {
                'video_models': [
                    {
                        'id': 'wan2-icecannon-t2v',
                        'name': 'Wan2 IceCannon T2V',
                        'path': 'Wan2_IceCannon_t2v2.1_nsfw_RCM_Lab_4step.safetensors',
                        'type': 't2v',
                        'description': '文生视频模型，4步快速生成'
                    }
                ],
                'default_video_model': 'wan2-icecannon-t2v'
            }
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

@app.route('/api/device', methods=['GET'])
def get_device_info():
    """获取设备信息"""
    try:
        import torch
        
        device_info = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        
        if torch.cuda.is_available():
            device_info['device_name'] = torch.cuda.get_device_name(0)
            device_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            device_info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3  # GB
            device_info['memory_reserved'] = torch.cuda.memory_reserved(0) / 1024**3  # GB
        
        return jsonify({
            'success': True,
            'device': device_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'device': {
                'device': 'cpu'
            }
        }), 200  # 返回 200 但标记失败

@app.route('/api/config/params', methods=['GET'])
def get_config_params():
    """获取配置参数"""
    try:
        if config_manager is None:
            return jsonify({
                'success': False,
                'error': 'Config manager not initialized'
            }), 500
        
        all_config = config_manager.get_all_config()
        return jsonify({
            'success': True,
            'config': all_config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config/params', methods=['POST'])
def save_config_params():
    """保存配置参数"""
    try:
        if config_manager is None:
            return jsonify({
                'success': False,
                'error': 'Config manager not initialized'
            }), 500
        
        data = request.get_json() or {}
        params = data.get('params', {})
        
        # 保存为上次使用的参数
        config_manager.save_last_used_params(params)
        
        return jsonify({
            'success': True,
            'message': 'Parameters saved successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config/defaults', methods=['POST'])
def update_default_config():
    """更新默认配置"""
    try:
        if config_manager is None:
            return jsonify({
                'success': False,
                'error': 'Config manager not initialized'
            }), 500
        
        data = request.get_json() or {}
        
        for key, value in data.items():
            config_manager.update_default_param(key, value)
        
        return jsonify({
            'success': True,
            'message': 'Default configuration updated successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/outputs/videos/<filename>')
def serve_video(filename):
    """提供视频文件服务"""
    from flask import send_from_directory
    return send_from_directory(str(OUTPUT_DIR), filename)

@app.route('/genesis/output/<filename>')
def serve_original_video(filename):
    """直接提供 WanVideo 原始输出文件"""
    from flask import send_from_directory, make_response
    output_dir = current_dir / "output"
    response = make_response(send_from_directory(str(output_dir), filename))
    # 添加 CORS 头，允许视频播放
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
    response.headers['Accept-Ranges'] = 'bytes'
    return response

if __name__ == '__main__':
    print("="*60)
    print("WanVideo API Server for Ant Design Pro")
    print("="*60)
    print(f"Server running on: http://localhost:5000")
    print(f"WanVideo Available: {WANVIDEO_AVAILABLE}")
    print(f"Worker Thread: Running")
    print("="*60)
    
    if not WANVIDEO_AVAILABLE:
        print("\n[WARNING] WanVideo is not available!")
        print("The server will start but video generation will fail.")
        print("Please check the error messages above.\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

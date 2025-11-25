"""
WanVideo Gradio Application
Text-to-Video and Image-to-Video generation using Genesis Core + ComfyUI-WanVideoWrapper
Version: 3.0.1 - Fixed I2V issues: scheduler list, InfiniteTalk logic, button click
"""

print("[INFO] Loading WanVideo Gradio App v3.0.1 - I2V Fixes")

import gradio as gr
import numpy as np
import torch
import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import glob
import soundfile as sf
from datetime import datetime

# Fix console encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to Python path
current_dir = Path(__file__).parent  # genesis/apps/
genesis_dir = current_dir.parent  # genesis/
project_root = genesis_dir.parent  # e:\Comfyu3.13---test\
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(genesis_dir))

# Apply Gradio API fix for boolean schema handling
try:
    from fix_gradio_api import *
except Exception:
    # Inline fallback patch (handles boolean schemas like additionalProperties: false)
    try:
        import gradio_client.utils as _gradio_utils
        _orig__json_schema_to_python_type = _gradio_utils._json_schema_to_python_type
        def _fixed__json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"
            if schema is None:
                return "None"
            return _orig__json_schema_to_python_type(schema, defs)
        _gradio_utils._json_schema_to_python_type = _fixed__json_schema_to_python_type
        print("[INFO] Applied Gradio API fix for boolean schema handling (inline)")
    except Exception as _e:
        print(f"[WARN] Could not apply Gradio API fix: {str(_e)}")

# Set up environment
os.environ['COMFYUI_PATH'] = str(project_root)

# Check for Sage Attention support (memory optimization)
try:
    import sageattention
    SAGE_ATTENTION_AVAILABLE = True
    print("[INFO] Sage Attention available - memory optimization enabled")
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False
    print("[INFO] Sage Attention not available - using standard attention")

# Import compatibility layers first
# 将 genesis 注册为模块
import importlib.util
genesis_init_path = genesis_dir / "__init__.py"
spec = importlib.util.spec_from_file_location("genesis", genesis_init_path)
if spec and spec.loader:
    genesis_module = importlib.util.module_from_spec(spec)
    sys.modules['genesis'] = genesis_module
    spec.loader.exec_module(genesis_module)

# Import triton stub before anything else
from genesis.utils import triton_ops_stub

# Import genesis components
from genesis.compat import comfy_stub
from genesis.core import folder_paths_ext

# Setup ComfyUI-WanVideoWrapper as a module
import importlib.util
wrapper_path = project_root / "genesis" / "custom_nodes" / "Comfyui" / "ComfyUI-WanVideoWrapper"
spec = importlib.util.spec_from_file_location(
    "ComfyUI_WanVideoWrapper",
    wrapper_path / "__init__.py"
)
wrapper_module = importlib.util.module_from_spec(spec)
sys.modules['ComfyUI_WanVideoWrapper'] = wrapper_module

# Execute the module to load all nodes
try:
    spec.loader.exec_module(wrapper_module)
    NODE_CLASS_MAPPINGS = getattr(wrapper_module, 'NODE_CLASS_MAPPINGS', {})
    NODE_DISPLAY_NAME_MAPPINGS = getattr(wrapper_module, 'NODE_DISPLAY_NAME_MAPPINGS', {})
    print(f"[INFO] Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes from ComfyUI-WanVideoWrapper")

    # Check for essential nodes
    essential_nodes = [
        'LoadWanVideoT5TextEncoder',
        'WanVideoTextEncode',
        'WanVideoModelLoader',
        'WanVideoVAELoader',
        'WanVideoEmptyEmbeds',
        'WanVideoSampler',
        'WanVideoDecode'
    ]

    missing_nodes = [node for node in essential_nodes if node not in NODE_CLASS_MAPPINGS]
    if missing_nodes:
        print(f"[WARNING] Missing essential nodes: {missing_nodes}")
    else:
        print(f"[INFO] All essential nodes loaded successfully")

except Exception as e:
    print(f"[ERROR] Failed to load ComfyUI-WanVideoWrapper: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


def load_audio_with_soundfile(audio_file):
    """
    使用 soundfile 加载音频文件（支持 MP3, WAV, FLAC 等）
    
    Args:
        audio_file: 音频文件路径
        
    Returns:
        tuple: (waveform_tensor, sample_rate) 或 (None, None) 如果失败
    """
    try:
        # 使用 soundfile 读取音频
        waveform_np, sample_rate = sf.read(audio_file, dtype='float32')
        
        # 转换为 torch tensor
        waveform = torch.from_numpy(waveform_np)
        
        # 确保是 (channels, samples) 格式 - 保持原来能工作的格式
        if waveform.dim() == 1:
            # 单声道: (samples,) -> (1, samples)
            waveform = waveform.unsqueeze(0)
        else:
            # 立体声: (samples, channels) -> (channels, samples)
            waveform = waveform.T
        
        print(f"[DEBUG] Audio loaded with soundfile: {sample_rate}Hz, shape={waveform.shape}")
        return waveform, sample_rate
        
    except Exception as e:
        print(f"[ERROR] soundfile failed to load audio: {e}")
        return None, None


class WanVideoWorkflow:
    """WanVideo workflow executor"""

    def __init__(self):
        # Get node mappings directly
        self.nodes = NODE_CLASS_MAPPINGS

        # Initialize node instances (check if nodes exist)
        self.t5_encoder = self.nodes.get("LoadWanVideoT5TextEncoder")() if self.nodes.get("LoadWanVideoT5TextEncoder") else None
        self.text_encoder = self.nodes.get("WanVideoTextEncode")() if self.nodes.get("WanVideoTextEncode") else None
        self.model_loader = self.nodes.get("WanVideoModelLoader")() if self.nodes.get("WanVideoModelLoader") else None
        self.vae_loader = self.nodes.get("WanVideoVAELoader")() if self.nodes.get("WanVideoVAELoader") else None
        self.empty_embeds = self.nodes.get("WanVideoEmptyEmbeds")() if self.nodes.get("WanVideoEmptyEmbeds") else None
        self.sampler = self.nodes.get("WanVideoSampler")() if self.nodes.get("WanVideoSampler") else None
        self.decoder = self.nodes.get("WanVideoDecode")() if self.nodes.get("WanVideoDecode") else None

        # Optimization nodes (optional)
        # 优先使用 Multi 版本支持多 LoRA，如果不存在则回退到单 LoRA 版本
        self.lora_selector_multi = self.nodes.get("WanVideoLoraSelectMulti")() if self.nodes.get("WanVideoLoraSelectMulti") else None
        self.lora_selector = self.nodes.get("WanVideoLoraSelect")() if self.nodes.get("WanVideoLoraSelect") else None
        self.compile_settings = self.nodes.get("WanVideoTorchCompileSettings")() if self.nodes.get("WanVideoTorchCompileSettings") else None
        self.block_swap = self.nodes.get("WanVideoBlockSwap")() if self.nodes.get("WanVideoBlockSwap") else None

        # Check required nodes
        if not all([self.t5_encoder, self.text_encoder, self.model_loader, self.vae_loader,
                    self.empty_embeds, self.sampler, self.decoder]):
            missing = []
            if not self.t5_encoder: missing.append("LoadWanVideoT5TextEncoder")
            if not self.text_encoder: missing.append("WanVideoTextEncode")
            if not self.model_loader: missing.append("WanVideoModelLoader")
            if not self.vae_loader: missing.append("WanVideoVAELoader")
            if not self.empty_embeds: missing.append("WanVideoEmptyEmbeds")
            if not self.sampler: missing.append("WanVideoSampler")
            if not self.decoder: missing.append("WanVideoDecode")
            print(f"Missing nodes: {missing}")
            print("Available nodes:", list(self.nodes.keys()))
            raise RuntimeError("Failed to initialize all required nodes")

        self.current_model = None
        self.current_vae = None
        self.current_t5 = None
        
        # ✅ 模型缓存机制
        self._model_cache = {}
        self._vae_cache = {}
        self._t5_cache = {}
        self._wav2vec_cache = {}
        self._text_embeds_cache = {}
        print("[INFO] Model caching system initialized")
    
    def _get_or_load_model(self, model_name, base_precision, quantization, load_device, attention_mode):
        """获取或加载 Diffusion Model（带缓存）"""
        cache_key = f"{model_name}_{base_precision}_{quantization}_{load_device}_{attention_mode}"
        
        if cache_key in self._model_cache:
            print(f"[CACHE] Using cached model: {model_name}")
            return self._model_cache[cache_key]
        
        print(f"[LOAD] Loading model: {model_name} (first time)")
        model_loader = NODE_CLASS_MAPPINGS['WanVideoModelLoader']()
        model_result = model_loader.loadmodel(
            model=model_name,
            base_precision=base_precision,
            quantization=quantization,
            load_device=load_device,
            attention_mode=attention_mode
        )
        self._model_cache[cache_key] = model_result[0]
        print(f"[CACHE] Model cached: {cache_key}")
        return self._model_cache[cache_key]
    
    def _get_or_load_vae(self, vae_name, precision):
        """获取或加载 VAE（带缓存）"""
        cache_key = f"{vae_name}_{precision}"
        
        if cache_key in self._vae_cache:
            print(f"[CACHE] Using cached VAE: {vae_name}")
            return self._vae_cache[cache_key]
        
        print(f"[LOAD] Loading VAE: {vae_name} (first time)")
        vae_loader = NODE_CLASS_MAPPINGS['WanVideoVAELoader']()
        vae_result = vae_loader.loadmodel(
            model_name=vae_name,
            precision=precision
        )
        self._vae_cache[cache_key] = vae_result[0]
        print(f"[CACHE] VAE cached: {cache_key}")
        return self._vae_cache[cache_key]
    
    def _get_or_load_t5(self, t5_model, precision, load_device, quantization):
        """获取或加载 T5（带缓存）"""
        cache_key = f"{t5_model}_{precision}_{load_device}_{quantization}"
        
        if cache_key in self._t5_cache:
            print(f"[CACHE] Using cached T5: {t5_model}")
            return self._t5_cache[cache_key]
        
        print(f"[LOAD] Loading T5: {t5_model} (first time)")
        t5_loader = NODE_CLASS_MAPPINGS['LoadWanVideoT5TextEncoder']()
        t5_result = t5_loader.loadmodel(
            model_name=t5_model,
            precision=precision,
            load_device=load_device,
            quantization=quantization
        )
        self._t5_cache[cache_key] = t5_result[0]
        print(f"[CACHE] T5 cached: {cache_key}")
        return self._t5_cache[cache_key]
    
    def _get_or_load_wav2vec(self, model_name, base_precision, load_device):
        """获取或加载 Wav2Vec（带缓存）"""
        cache_key = f"{model_name}_{base_precision}_{load_device}"
        
        if cache_key in self._wav2vec_cache:
            print(f"[CACHE] Using cached Wav2Vec: {model_name}")
            return self._wav2vec_cache[cache_key]
        
        print(f"[LOAD] Loading Wav2Vec: {model_name} (first time)")
        wav2vec_loader = NODE_CLASS_MAPPINGS['DownloadAndLoadWav2VecModel']()
        wav2vec_result = wav2vec_loader.loadmodel(
            model=model_name,
            base_precision=base_precision,
            load_device=load_device
        )
        self._wav2vec_cache[cache_key] = wav2vec_result[0]
        print(f"[CACHE] Wav2Vec cached: {cache_key}")
        return self._wav2vec_cache[cache_key]

    def generate_image_to_video(
        self,
        # Input image
        input_image,
        # Text prompts
        positive_prompt: str,
        negative_prompt: str,
        # Model selection
        model_name: str,
        vae_name: str,
        t5_model: str,
        # Generation parameters
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        shift: float,
        seed: int,
        scheduler: str,
        denoise_strength: float,
        # Model config
        base_precision: str,
        quantization: str,
        attention_mode: str,
        # Mode and mode-specific parameters
        mode: str = "Standard I2V",
        audio_file: Optional[str] = None,
        frame_window_size: int = 117,
        motion_frame: int = 25,
        # Wav2Vec parameters
        wav2vec_precision: str = "fp16",
        wav2vec_device: str = "main_device",
        # Image processing parameters
        keep_proportion: str = "crop",
        crop_position: str = "center",
        upscale_method: str = "lanczos",
        pose_images = None,
        face_images = None,
        pose_strength: float = 1.0,
        face_strength: float = 1.0,
        colormatch: str = 'mkl',
        # Output parameters
        fps: int = 25,
        output_format: str = "mp4",
        # LoRA parameters
        lora_enabled: bool = False,
        lora_name: str = "",
        lora_strength: float = 1.0,
        # Optimization parameters
        compile_enabled: bool = False,
        compile_backend: str = "inductor",
        block_swap_enabled: bool = False,
        # VRAM Management parameters
        auto_hardware_tuning: bool = True,
        vram_threshold_percent: float = 50.0,
        blocks_to_swap: int = 0,
        enable_cuda_optimization: bool = True,
        enable_dram_optimization: bool = True,
        num_cuda_streams: int = 8,
        bandwidth_target: float = 0.8,
        offload_txt_emb: bool = False,
        offload_img_emb: bool = False,
        vace_blocks_to_swap: int = 0,
        vram_debug_mode: bool = False,
        progress_callback=None
    ):
        """Execute image to video generation workflow"""
        
        print("\n" + "="*60)
        print(f"Starting Image to Video Generation - Mode: {mode}")
        print("="*60)
        print(f"Prompt: {positive_prompt[:100]}...")
        print(f"Model: {model_name}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        print(f"Scheduler: {scheduler}")
        print("="*60)
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "parameters": {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "steps": steps,
                "cfg": cfg,
                "shift": shift,
                "seed": seed,
                "scheduler": scheduler,
            }
        }
        
        try:
            # ✅ 使用缓存加载模型
            print("[INFO] Loading models...")
            if progress_callback:
                progress_callback(0.05, "Loading models...")
            
            # Load model with caching
            load_device = "main_device" if mode == "InfiniteTalk" else "offload_device"
            model = self._get_or_load_model(
                model_name=model_name,
                base_precision=base_precision,
                quantization=quantization,
                load_device=load_device,
                attention_mode=attention_mode
            )
            
            if progress_callback:
                progress_callback(0.15, "Loading VAE...")
            
            # Load VAE with caching
            vae = self._get_or_load_vae(
                vae_name=vae_name,
                precision="bf16"
            )
            
            if progress_callback:
                progress_callback(0.25, "Loading T5 encoder...")
            
            # Load T5 with caching
            t5_encoder = self._get_or_load_t5(
                t5_model=t5_model,
                precision="bf16",
                load_device="offload_device",
                quantization="disabled"
            )
            
            if progress_callback:
                progress_callback(0.35, "Encoding text...")
            
            # Encode text
            print("[INFO] Encoding text...")
            text_encoder = NODE_CLASS_MAPPINGS['WanVideoTextEncode']()
            text_embeds = text_encoder.process(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                t5=t5_encoder,
                force_offload=True,
                use_disk_cache=True,  # ✅ 启用缓存
                device="gpu"
            )[0]
            print(f"[INFO] Text encoded successfully")
            
            # Process input image
            print("[INFO] Processing input image...")
            import numpy as np
            from PIL import Image
            
            if isinstance(input_image, Image.Image):
                # Convert PIL to tensor
                img_array = np.array(input_image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            else:
                img_tensor = input_image
            
            # Create image embeds based on mode
            if mode == "InfiniteTalk":
                print("[INFO] Using InfiniteTalk mode...")
                print(f"[DEBUG] Image tensor shape: {img_tensor.shape}")
                
                if progress_callback:
                    progress_callback(0.45, "Processing InfiniteTalk...")
                
                # Load MultiTalk/InfiniteTalk specific nodes
                print("[DEBUG] Loading WanVideoImageToVideoMultiTalk node...")
                multitalk_i2v_node = NODE_CLASS_MAPPINGS['WanVideoImageToVideoMultiTalk']()
                print("[DEBUG] WanVideoImageToVideoMultiTalk node loaded")
                
                # InfiniteTalk doesn't use CLIP Vision in the workflow
                # It uses direct image embedding through the MultiTalk node
                clip_embeds = None
                print("[INFO] InfiniteTalk mode: using direct image embedding (no CLIP Vision needed)")
                
                # Process audio if provided, otherwise create silent embeds
                print("[DEBUG] Checking for audio file...")
                audio_embeds = None
                if audio_file is not None and audio_file != "":
                    print(f"[INFO] Loading audio: {audio_file}")
                    try:
                        # ✅ 使用缓存加载 Wav2Vec 模型
                        wav2vec_model_name = "TencentGameMate/chinese-wav2vec2-base"
                        print(f"[DEBUG] Using Wav2Vec model: {wav2vec_model_name}")
                        print(f"[DEBUG] Wav2Vec config: precision={wav2vec_precision}, device={wav2vec_device}")
                        
                        wav2vec_model = self._get_or_load_wav2vec(
                            model_name=wav2vec_model_name,
                            base_precision=wav2vec_precision,
                            load_device=wav2vec_device
                        )
                        
                        # Load audio file using soundfile
                        print(f"[DEBUG] Loading audio file...")
                        waveform, sample_rate = load_audio_with_soundfile(audio_file)
                        
                        if waveform is None:
                            print(f"[WARNING] Could not load audio file, using silent mode...")
                            audio_embeds = None
                        else:
                            # 添加 batch 维度以匹配 ComfyUI AUDIO 格式 (batch, channels, samples)
                            audio_data = {
                                "waveform": waveform.unsqueeze(0),  # (channels, samples) -> (1, channels, samples)
                                "sample_rate": sample_rate
                            }
                            print(f"[DEBUG] Audio file loaded: sample_rate={sample_rate}, shape={waveform.shape}")
                            
                            # Create audio embeds
                            print("[DEBUG] Creating audio embeds...")
                            wav2vec_embeds_node = NODE_CLASS_MAPPINGS['MultiTalkWav2VecEmbeds']()
                            audio_embeds_result = wav2vec_embeds_node.process(
                                wav2vec_model=wav2vec_model,
                                audio_1=audio_data,
                                normalize_loudness=True,
                                num_frames=frame_window_size,
                                fps=fps,
                                audio_scale=1.0,
                                audio_cfg_scale=1.0,
                                multi_audio_type="para"
                            )
                            audio_embeds = audio_embeds_result[0]
                            actual_num_frames = audio_embeds_result[2]
                            print(f"[INFO] Audio embeds created, actual frames: {actual_num_frames}")
                    except Exception as e:
                        print(f"[WARNING] Audio processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                        print("[INFO] Continuing without audio...")
                        audio_embeds = None
                
                # If no audio or audio failed, create silent embeds for InfiniteTalk
                if audio_embeds is None:
                    print("[DEBUG] No audio provided, creating silent embeds for InfiniteTalk...")
                    try:
                        silent_embeds_node = NODE_CLASS_MAPPINGS['MultiTalkSilentEmbeds']()
                        silent_result = silent_embeds_node.process(num_frames=frame_window_size)
                        audio_embeds = silent_result[0]
                        print(f"[INFO] Silent embeds created for {frame_window_size} frames")
                    except Exception as silent_error:
                        print(f"[ERROR] Failed to create silent embeds: {silent_error}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                # Create image embeds using MultiTalk I2V node
                # Reference workflow: [832, 480, 117, 25, False, 'mkl', False, 'infinitetalk', '']
                print("[DEBUG] Creating InfiniteTalk image embeds...")
                print(f"[DEBUG] Parameters: width={width}, height={height}, frame_window={frame_window_size}, motion_frame={motion_frame}")
                try:
                    # Enable memory optimizations
                    use_tiled_vae = True  # Reduce VAE memory usage
                    use_force_offload = True  # Enable model offloading
                    print(f"[DEBUG] Memory optimizations: tiled_vae={use_tiled_vae}, force_offload={use_force_offload}")
                    
                    result = multitalk_i2v_node.process(
                        vae=vae,
                        width=width,
                        height=height,
                        frame_window_size=frame_window_size,
                        motion_frame=motion_frame,
                        force_offload=use_force_offload,  # Enable offload for memory saving
                        colormatch=colormatch,
                        start_image=img_tensor,
                        tiled_vae=use_tiled_vae,  # Enable tiled VAE for memory saving
                        clip_embeds=clip_embeds,
                        mode='infinitetalk',
                        output_path=""
                    )
                    print(f"[DEBUG] MultiTalk node result type: {type(result)}")
                    print(f"[DEBUG] MultiTalk node result length: {len(result) if result else 'None'}")
                    
                    if result is None:
                        raise ValueError("MultiTalk node returned None")
                    
                    image_embeds = result[0]
                    print(f"[DEBUG] Image embeds type: {type(image_embeds)}")
                    if isinstance(image_embeds, dict):
                        print(f"[DEBUG] Image embeds keys: {list(image_embeds.keys())}")
                        # Check critical keys
                        if 'multitalk_sampling' in image_embeds:
                            print(f"[DEBUG] multitalk_sampling: {image_embeds['multitalk_sampling']}")
                        if 'multitalk_start_image' in image_embeds:
                            start_img = image_embeds['multitalk_start_image']
                            print(f"[DEBUG] multitalk_start_image: {type(start_img)}, shape: {start_img.shape if start_img is not None else 'None'}")
                        if 'vae' in image_embeds:
                            print(f"[DEBUG] VAE in image_embeds: {type(image_embeds['vae'])}")
                    else:
                        print(f"[DEBUG] Image embeds is not a dict: {image_embeds}")
                    print(f"[INFO] InfiniteTalk embeds created successfully with frame_window={frame_window_size}, motion_frame={motion_frame}")
                except Exception as embed_error:
                    print(f"[ERROR] Failed to create InfiniteTalk embeds: {embed_error}")
                    import traceback
                    traceback.print_exc()
                    raise
                
            elif mode == "WanAnimate":
                print("[INFO] Using WanAnimate mode...")
                animate_embeds_node = NODE_CLASS_MAPPINGS['WanVideoAnimateEmbeds']()
                
                # Prepare optional inputs
                pose_imgs = None
                face_imgs = None
                
                if pose_images is not None:
                    print("[INFO] Processing pose images...")
                    if isinstance(pose_images, Image.Image):
                        pose_array = np.array(pose_images).astype(np.float32) / 255.0
                        pose_imgs = torch.from_numpy(pose_array).unsqueeze(0)
                    else:
                        pose_imgs = pose_images
                
                if face_images is not None:
                    print("[INFO] Processing face images...")
                    if isinstance(face_images, Image.Image):
                        face_array = np.array(face_images).astype(np.float32) / 255.0
                        face_imgs = torch.from_numpy(face_array).unsqueeze(0)
                    else:
                        face_imgs = face_images
                
                # 参考工作流: WanVideoAnimateEmbeds
                # widgets: [832, 480, 81, True, 77, 'disabled', 1, 1, False]
                image_embeds = animate_embeds_node.process(
                    vae=vae,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    force_offload=True,
                    frame_window_size=frame_window_size,
                    colormatch=colormatch,
                    pose_strength=pose_strength,
                    face_strength=face_strength,
                    tiled_vae=False,
                    ref_images=img_tensor,
                    pose_images=pose_imgs,
                    face_images=face_imgs
                )[0]
                print("[INFO] Animate embeds created")
                
            else:  # Standard I2V
                print("[INFO] Using Standard I2V mode...")
                i2v_embeds_node = NODE_CLASS_MAPPINGS['WanVideoImageToVideoEncode']()
                image_embeds = i2v_embeds_node.process(
                    vae=vae,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    force_offload=True,
                    noise_aug_strength=0.0,
                    start_latent_strength=1.0,
                    end_latent_strength=1.0,
                    start_image=img_tensor,
                    end_image=None,
                    add_cond_latents=None
                )[0]
            
            # Sample
            if progress_callback:
                progress_callback(0.55, "Starting sampling...")
            
            print("[INFO] Starting sampling...")
            sampler = NODE_CLASS_MAPPINGS['WanVideoSampler']()
            
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            # Determine scheduler
            # Note: multitalk scheduler only works with multitalk_sampling mode
            # For InfiniteTalk, use regular schedulers like unipc or dpm++
            actual_scheduler = scheduler  # Don't force multitalk scheduler
            
            # Print debug info
            print(f"[DEBUG] Steps requested: {steps}")
            print(f"[DEBUG] CFG: {cfg}")
            print(f"[DEBUG] Shift: {shift}")
            print(f"[DEBUG] Scheduler: {actual_scheduler}")
            print(f"[DEBUG] Seed: {seed}")
            
            # Prepare sampler arguments - match T2V implementation
            # Reference: T2V uses positional order matching WanVideoSampler.process signature
            print(f"[INFO] Starting sampling with {steps} steps using {actual_scheduler} scheduler...")
            print(f"[DEBUG] Model type: {type(model)}")
            print(f"[DEBUG] Image embeds type: {type(image_embeds)}")
            
            # Validate image_embeds
            if image_embeds is None:
                raise ValueError("image_embeds is None before sampling")
            
            if isinstance(image_embeds, dict):
                print(f"[DEBUG] Image embeds is dict with keys: {list(image_embeds.keys())}")
                if "multitalk_sampling" in image_embeds:
                    print(f"[DEBUG] multitalk_sampling = {image_embeds['multitalk_sampling']}")
            elif isinstance(image_embeds, tuple):
                print(f"[DEBUG] Image embeds is tuple with length: {len(image_embeds)}")
                if len(image_embeds) > 0 and image_embeds[0] is not None:
                    print(f"[DEBUG] Image embeds[0] shape: {image_embeds[0].shape}")
            else:
                print(f"[DEBUG] Image embeds is: {image_embeds}")
            
            # Validate text_embeds
            print(f"[DEBUG] Text embeds type: {type(text_embeds)}")
            if text_embeds is not None:
                if isinstance(text_embeds, dict):
                    print(f"[DEBUG] Text embeds is dict with keys: {list(text_embeds.keys())}")
                elif hasattr(text_embeds, 'shape'):
                    print(f"[DEBUG] Text embeds shape: {text_embeds.shape}")
            
            # Build sampler arguments matching T2V implementation
            sampler_args = {
                "model": model,
                "image_embeds": image_embeds,
                "steps": steps,
                "cfg": cfg,
                "shift": shift,
                "seed": seed,
                "scheduler": actual_scheduler,
                "riflex_freq_index": 0,
                "force_offload": True,
                "text_embeds": text_embeds
            }
            
            # Add mode-specific parameters
            if mode == "InfiniteTalk":
                # InfiniteTalk always needs multitalk_embeds (either real audio or silent)
                sampler_args["multitalk_embeds"] = audio_embeds
                if audio_file is not None and audio_file != "":
                    print("[INFO] Using real audio embeds for InfiniteTalk")
                else:
                    print("[INFO] Using silent embeds for InfiniteTalk (no audio provided)")
            
            if denoise_strength < 1.0:
                sampler_args["denoise_strength"] = denoise_strength
            
            print(f"[DEBUG] Sampler args keys: {list(sampler_args.keys())}")
            
            # Add a simple progress monitor
            import sys
            sys.stdout.flush()
            
            try:
                print(f"[DEBUG] Calling sampler.process()...")
                result = sampler.process(**sampler_args)
                print(f"[DEBUG] Sampler result type: {type(result)}")
                print(f"[DEBUG] Sampler result length: {len(result) if result else 'None'}")
                
                if result is None:
                    raise ValueError("Sampler returned None")
                
                samples = result[0]
                print(f"[INFO] Sampling completed successfully")
            except Exception as sample_error:
                print(f"[ERROR] Sampling failed: {sample_error}")
                import traceback
                traceback.print_exc()
                raise
            
            # Decode
            if progress_callback:
                progress_callback(0.85, "Decoding video...")
            
            print("[INFO] Decoding video...")
            decoder = NODE_CLASS_MAPPINGS['WanVideoDecode']()
            video_result = decoder.decode(
                vae=vae,
                samples=samples,
                enable_vae_tiling=False,
                tile_x=272,
                tile_y=272,
                tile_stride_x=144,
                tile_stride_y=128,
                normalization="default"
            )
            
            video_tensor = video_result[0]
            print(f"[INFO] Video decoded successfully, shape: {video_tensor.shape}")
            
            # Convert to numpy
            if progress_callback:
                progress_callback(0.95, "Converting to video...")
            
            if isinstance(video_tensor, torch.Tensor):
                # video_tensor shape: [frames, height, width, channels]
                video_array = (video_tensor.cpu().numpy() * 255).astype(np.uint8)
                print(f"[INFO] Video array shape: {video_array.shape}")
            else:
                video_array = video_tensor
            
            # Save video
            output_dir = Path("outputs/i2v")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = output_dir / f"i2v_{mode.lower().replace(' ', '_')}_{timestamp}.mp4"
            
            print(f"[INFO] Saving video to: {video_path}")
            print(f"[INFO] Video shape: {video_array.shape}")
            print(f"[INFO] FPS: {fps}")
            
            try:
                import subprocess
                
                # 检查是否需要添加音频（InfiniteTalk 模式且有音频文件）
                if mode == "InfiniteTalk" and audio_file and audio_file != "":
                    print(f"[INFO] InfiniteTalk mode with audio detected")
                    
                    # 1. 先保存无声视频（使用 cv2，更可靠）
                    video_path_no_audio = output_dir / f"infinitetalk_{timestamp}_no_audio.mp4"
                    print(f"[INFO] Saving temporary video (no audio) using cv2...")
                    
                    try:
                        import cv2
                        height, width = video_array.shape[1:3]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(video_path_no_audio), fourcc, fps, (width, height))
                        
                        for frame in video_array:
                            # Convert RGB to BGR for cv2
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)
                        
                        out.release()
                        print(f"[INFO] Temporary video (no audio) saved: {video_path_no_audio}")
                    except Exception as cv2_error:
                        print(f"[ERROR] cv2 failed: {cv2_error}")
                        # 尝试使用 imageio
                        try:
                            import imageio
                            imageio.mimwrite(str(video_path_no_audio), video_array, fps=fps, quality=8, codec='libx264')
                            print(f"[INFO] Temporary video saved with imageio")
                        except Exception as imageio_error:
                            print(f"[ERROR] imageio also failed: {imageio_error}")
                            raise Exception("Both cv2 and imageio failed to save video")
                    
                    # 2. 使用 ffmpeg 合并音频
                    video_path_with_audio = output_dir / f"infinitetalk_{timestamp}.mp4"
                    
                    print(f"[INFO] Merging audio with video using ffmpeg...")
                    print(f"[DEBUG] Video: {video_path_no_audio}")
                    print(f"[DEBUG] Audio: {audio_file}")
                    print(f"[DEBUG] Output: {video_path_with_audio}")
                    
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', str(video_path_no_audio),  # 输入视频
                        '-i', audio_file,                 # 输入音频
                        '-c:v', 'copy',                   # 复制视频流（不重新编码，快速）
                        '-c:a', 'aac',                    # 音频编码为 AAC
                        '-b:a', '192k',                   # 音频比特率
                        '-shortest',                      # 使用较短的流长度（视频或音频）
                        str(video_path_with_audio)
                    ]
                    
                    try:
                        result = subprocess.run(
                            cmd, 
                            check=True, 
                            capture_output=True, 
                            text=True,
                            encoding='utf-8',    # 使用 UTF-8 编码避免 GBK 解码错误
                            errors='replace'     # 遇到无法解码的字符替换为 �，不会崩溃
                        )
                        print(f"[SUCCESS] Video with audio saved to: {video_path_with_audio}")
                        
                        # 删除临时无声视频
                        try:
                            video_path_no_audio.unlink()
                            print(f"[INFO] Temporary video deleted")
                        except Exception as del_error:
                            print(f"[WARNING] Failed to delete temporary video: {del_error}")
                        
                        # 使用带音频的视频路径
                        video_path = video_path_with_audio
                        
                    except subprocess.CalledProcessError as e:
                        print(f"[WARNING] Failed to merge audio with ffmpeg:")
                        print(f"[WARNING] Error: {e.stderr}")
                        print(f"[INFO] Falling back to video without audio")
                        video_path = video_path_no_audio
                    except FileNotFoundError:
                        print(f"[ERROR] ffmpeg not found! Please install ffmpeg.")
                        print(f"[INFO] Using video without audio: {video_path_no_audio}")
                        video_path = video_path_no_audio
                else:
                    # 没有音频或不是 InfiniteTalk 模式，直接保存
                    print(f"[INFO] Saving video using cv2...")
                    try:
                        import cv2
                        height, width = video_array.shape[1:3]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                        
                        for frame in video_array:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)
                        
                        out.release()
                        print(f"[SUCCESS] Video saved successfully to: {video_path}")
                    except Exception as cv2_error:
                        print(f"[WARNING] cv2 failed: {cv2_error}, trying imageio...")
                        try:
                            import imageio
                            imageio.mimwrite(str(video_path), video_array, fps=fps, quality=8, codec='libx264')
                            print(f"[SUCCESS] Video saved with imageio: {video_path}")
                        except Exception as imageio_error:
                            print(f"[ERROR] imageio also failed: {imageio_error}")
                            raise Exception("Both cv2 and imageio failed to save video")
                    
            except Exception as save_error:
                print(f"[ERROR] Failed to save video: {save_error}")
                import traceback
                traceback.print_exc()
                # Try alternative save method
                try:
                    import cv2
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    height, width = video_array.shape[1:3]
                    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                    for frame in video_array:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
                    print(f"[SUCCESS] Video saved using cv2: {video_path}")
                except Exception as cv2_error:
                    print(f"[ERROR] cv2 save also failed: {cv2_error}")
                    raise
            
            print("="*60)
            
            metadata["output_path"] = str(video_path)
            metadata["seed_used"] = seed
            metadata["actual_frames"] = len(video_array)
            
            return str(video_path), video_array, metadata
            
        except Exception as e:
            print(f"[ERROR] Image to video generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def generate_video(
        self,
        # Text parameters
        positive_prompt: str,
        negative_prompt: str,
        # Model parameters
        model_name: str,
        vae_name: str,
        t5_model: str,
        # Generation parameters
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        shift: float,
        seed: int,
        # Advanced parameters
        scheduler: str,
        denoise_strength: float,
        base_precision: str,
        quantization: str,
        attention_mode: str,
        # LoRA parameters
        lora_enabled: bool,
        lora_name: str,
        lora_strength: float,
        # Optimization parameters
        compile_enabled: bool,
        compile_backend: str,
        block_swap_enabled: bool,
        # Output parameters
        output_format: str,
        fps: int,
        # Intelligent VRAM Management parameters
        loras_list: list = None,  # 多 LoRA 支持
        lora_low_mem_load: bool = False,
        lora_merge_loras: bool = False,
        auto_hardware_tuning: bool = True,
        vram_threshold_percent: float = 50.0,
        blocks_to_swap: int = 0,
        enable_cuda_optimization: bool = True,
        enable_dram_optimization: bool = True,
        num_cuda_streams: int = 8,
        bandwidth_target: float = 0.8,
        offload_txt_emb: bool = False,
        offload_img_emb: bool = False,
        vace_blocks_to_swap: int = 0,
        vram_debug_mode: bool = False,
        progress_callback=None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute the video generation workflow"""

        print("\n" + "="*60)
        print("Starting WanVideo Generation")
        print("="*60)
        print(f"Prompt: {positive_prompt[:100]}...")
        print(f"Model: {model_name}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        print("="*60)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "parameters": locals().copy()
        }
        del metadata["parameters"]["self"]
        del metadata["parameters"]["progress_callback"]

        try:
            if progress_callback:
                progress_callback(0.05, "Loading T5 text encoder...")

            # Step 1: Load T5 encoder
            print(f"\n[Step 1] Loading T5 encoder: {t5_model}")
            t5_result = self.t5_encoder.loadmodel(
                model_name=t5_model,
                precision="bf16",  # Use bf16 for precision
                quantization="fp8_e4m3fn"  # Use fp8_e4m3fn for quantization
            )
            t5_encoder = t5_result[0] if t5_result else None
            print(f"  [OK] T5 encoder loaded")

            if progress_callback:
                progress_callback(0.1, "Encoding text prompts...")

            # Step 2: Encode text
            print("="*60)
            print("[DEBUG] *** TEXT ENCODING START ***")
            print("="*60)
            print(f"[DEBUG] Encoding text with T5 encoder: {type(t5_encoder)}")
            print(f"[DEBUG] Positive prompt: {positive_prompt[:50]}...")
            print(f"[DEBUG] Negative prompt: {negative_prompt[:50] if negative_prompt else 'None'}...")
            text_embeds_result = self.text_encoder.process(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt if negative_prompt else "",
                t5=t5_encoder,
                force_offload=True,
                use_disk_cache=False,
                device="gpu"
            )
            print(f"[DEBUG] Text encode result: {type(text_embeds_result)}, length: {len(text_embeds_result) if text_embeds_result else 0}")
            text_embeds = text_embeds_result[0] if text_embeds_result else None
            print(f"[DEBUG] Text embeds extracted: {type(text_embeds)}")
            if text_embeds is not None and hasattr(text_embeds, 'shape'):
                print(f"[DEBUG] Text embeds shape: {text_embeds.shape}")
            elif text_embeds is not None:
                print(f"[DEBUG] Text embeds type (not tensor): {type(text_embeds)}, value: {text_embeds}")

            if progress_callback:
                progress_callback(0.15, "Setting up compile optimization...")

            # Step 3: Setup compile args if enabled
            compile_args = None
            if compile_enabled and self.compile_settings:
                try:
                    compile_result = self.compile_settings.prepare(
                        backend=compile_backend,
                        fullgraph=False,
                        mode="default",
                        dynamic=False,
                        cache_size=64,
                        enabled=True,
                        batch_size=128
                    )
                    compile_args = compile_result[0] if compile_result else None
                except:
                    compile_args = None

            if progress_callback:
                progress_callback(0.2, "Setting up intelligent VRAM management...")

            # Step 4: Setup Intelligent VRAM Management if enabled
            block_swap_args = None
            if block_swap_enabled:
                try:
                    from .intelligent_vram_manager import calculate_optimal_blockswap_config, get_vram_manager
                    
                    # 获取 VRAM 管理器
                    vram_manager = get_vram_manager(vram_threshold_percent)
                    
                    # 获取内存统计
                    stats = vram_manager.get_memory_stats()
                    
                    if vram_debug_mode:
                        print(f"\n[VRAM] 当前状态:")
                        print(f"  VRAM: {stats.vram_used_mb:.1f}/{stats.vram_total_mb:.1f}MB ({stats.vram_usage_percent:.1f}%)")
                        print(f"  DRAM: {stats.dram_used_mb:.1f}/{stats.dram_total_mb:.1f}MB ({stats.dram_usage_percent:.1f}%)")
                    
                    # 计算最优配置
                    if auto_hardware_tuning:
                        # 自动模式: 根据硬件自动计算
                        optimal_config = calculate_optimal_blockswap_config(
                            model_size_mb=0,  # 将在模型加载后更新
                            num_layers=24,
                            vram_threshold=vram_threshold_percent,
                            auto_hardware_tuning=True
                        )
                        
                        # 使用自动计算的参数
                        block_swap_args = {
                            "blocks_to_swap": optimal_config["blocks_to_swap"],
                            "num_cuda_streams": optimal_config["num_cuda_streams"],
                            "bandwidth_target": optimal_config["bandwidth_target"],
                            "enable_cuda_optimization": enable_cuda_optimization,
                            "enable_dram_optimization": enable_dram_optimization,
                            "vram_threshold_percent": vram_threshold_percent,
                            "offload_txt_emb": offload_txt_emb,
                            "offload_img_emb": offload_img_emb,
                            "vace_blocks_to_swap": vace_blocks_to_swap,
                        }
                        
                        if vram_debug_mode:
                            print(f"\n[VRAM] 自动配置:")
                            print(f"  分块数: {optimal_config['blocks_to_swap']}")
                            print(f"  CUDA 流: {optimal_config['num_cuda_streams']}")
                            print(f"  带宽目标: {optimal_config['bandwidth_target']:.0%}")
                    else:
                        # 手动模式: 使用用户指定的参数
                        block_swap_args = {
                            "blocks_to_swap": blocks_to_swap,
                            "num_cuda_streams": num_cuda_streams,
                            "bandwidth_target": bandwidth_target,
                            "enable_cuda_optimization": enable_cuda_optimization,
                            "enable_dram_optimization": enable_dram_optimization,
                            "vram_threshold_percent": vram_threshold_percent,
                            "offload_txt_emb": offload_txt_emb,
                            "offload_img_emb": offload_img_emb,
                            "vace_blocks_to_swap": vace_blocks_to_swap,
                        }
                        
                        if vram_debug_mode:
                            print(f"\n[VRAM] 手动配置:")
                            print(f"  分块数: {blocks_to_swap}")
                            print(f"  CUDA 流: {num_cuda_streams}")
                            print(f"  带宽目标: {bandwidth_target:.0%}")
                    
                    print(f"✓ 智能 VRAM 管理已启用 (阈值: {vram_threshold_percent}%)")
                    
                except Exception as e:
                    print(f"[ERROR] 智能 VRAM 管理初始化失败: {e}")
                    if vram_debug_mode:
                        import traceback
                        traceback.print_exc()
                    block_swap_args = None

            if progress_callback:
                progress_callback(0.25, "Loading LoRA...")

            # Step 5: Setup LoRA if enabled
            lora = None
            # 确保 loras_list 不为 None
            if loras_list is None:
                loras_list = []
            
            print(f"[DEBUG] LoRA check: enabled={lora_enabled}, loras_list={loras_list}")
            
            # 检查是否有 LoRA 列表（多 LoRA 支持）
            if loras_list and len(loras_list) > 0 and self.lora_selector_multi:
                try:
                    print(f"[INFO] Loading {len(loras_list)} LoRA(s) using WanVideoLoraSelectMulti")
                    
                    # 准备 LoRA 参数（最多10个）
                    lora_params = {}
                    for i in range(10):
                        if i < len(loras_list):
                            lora_params[f'lora_{i}'] = loras_list[i]['name']
                            lora_params[f'strength_{i}'] = loras_list[i]['strength']
                            print(f"  - LoRA {i}: {loras_list[i]['name']} (strength: {loras_list[i]['strength']})")
                        else:
                            lora_params[f'lora_{i}'] = 'none'
                            lora_params[f'strength_{i}'] = 1.0
                    
                    print(f"[INFO] LoRA settings: low_mem_load={lora_low_mem_load}, merge_loras={lora_merge_loras}")
                    lora_result = self.lora_selector_multi.getlorapath(
                        **lora_params,
                        blocks={},
                        prev_lora=None,
                        low_mem_load=lora_low_mem_load,
                        merge_loras=lora_merge_loras
                    )
                    lora = lora_result[0] if lora_result else None
                    if lora:
                        print(f"[INFO] Multi-LoRA loaded successfully: {len(lora)} LoRA(s)")
                    else:
                        print(f"[WARNING] Multi-LoRA loading returned None")
                except Exception as e:
                    print(f"[ERROR] Failed to load multi-LoRA: {e}")
                    import traceback
                    traceback.print_exc()
                    lora = None
            
            # 回退到单 LoRA 模式（兼容旧代码）
            elif lora_enabled and lora_name and self.lora_selector:
                try:
                    print(f"[INFO] Loading single LoRA: {lora_name} (strength: {lora_strength})")
                    print(f"[INFO] LoRA settings: low_mem_load={lora_low_mem_load}, merge_loras={lora_merge_loras}")
                    lora_result = self.lora_selector.getlorapath(
                        lora=lora_name,
                        strength=lora_strength,
                        unique_id="api_lora",
                        blocks={},
                        prev_lora=None,
                        low_mem_load=lora_low_mem_load,
                        merge_loras=lora_merge_loras
                    )
                    lora = lora_result[0] if lora_result else None
                    if lora:
                        print(f"[INFO] Single LoRA loaded successfully: {len(lora)} LoRA(s) in chain")
                    else:
                        print(f"[WARNING] Single LoRA loading returned None")
                except Exception as e:
                    print(f"[ERROR] Failed to load single LoRA: {e}")
                    import traceback
                    traceback.print_exc()
                    lora = None

            if progress_callback:
                progress_callback(0.3, "Loading main model...")

            # Step 6: Load main model with optimizations
            print(f"[DEBUG] Loading model: {model_name}")
            print(f"[DEBUG] Base precision: {base_precision}")
            print(f"[DEBUG] Quantization: {quantization}")
            print(f"[DEBUG] Attention mode: {attention_mode}")
            
            if lora:
                print(f"[INFO] Loading model with {len(lora)} LoRA(s)")
                print(f"[DEBUG] LoRA details:")
                for i, l in enumerate(lora):
                    print(f"  [{i}] Name: {l.get('name', 'unknown')}")
                    print(f"      Path: {l.get('path', 'unknown')}")
                    print(f"      Strength: {l.get('strength', 1.0)}")
                    print(f"      Merge: {l.get('merge_loras', True)}")
            
            # 清理显存，避免 OOM
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[DEBUG] CUDA cache cleared before model loading")
            
            print(f"[DEBUG] Calling loadmodel...")
            model_result = self.model_loader.loadmodel(
                model=model_name,
                base_precision=base_precision,
                quantization=quantization,
                load_device="offload_device",
                attention_mode=attention_mode,
                lora=lora  # 标准方式：通过 loadmodel 传递 LoRA
            )
            print(f"[DEBUG] loadmodel returned")
            model = model_result[0] if model_result else None
            print(f"[DEBUG] Model loaded: {type(model)}")
            
            if model and lora:
                has_patches = hasattr(model, 'patches') and len(model.patches) > 0
                if has_patches:
                    print(f"[INFO] LoRA applied as patches (unmerged mode): {len(model.patches)} patches")
                else:
                    print(f"[INFO] LoRA merged into model weights (merged mode)")
                    print(f"[SUCCESS] {len(lora)} LoRA(s) successfully merged into model")

            # Apply optimizations to model if available
            if model and compile_enabled:
                try:
                    # Apply torch.compile optimization to the model
                    import torch
                    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                        # Compile the diffusion model with specified backend
                        model.model.diffusion_model = torch.compile(
                            model.model.diffusion_model,
                            mode="default",
                            backend=compile_backend,
                            fullgraph=False
                        )
                        print(f"✓ Applied torch.compile with backend: {compile_backend}")
                except Exception as e:
                    print(f"Warning: Could not apply torch.compile: {e}")

            if model and block_swap_enabled:
                try:
                    # Apply block swap optimization for memory management
                    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                        # Set up block swapping for memory optimization
                        diffusion_model = model.model.diffusion_model
                        if hasattr(diffusion_model, 'blocks_to_swap'):
                            diffusion_model.blocks_to_swap = blocks_to_swap
                            print(f"✓ Block swap enabled with {blocks_to_swap} blocks")
                        # Additional memory optimization settings
                        if hasattr(diffusion_model, 'enable_memory_efficient_attention'):
                            diffusion_model.enable_memory_efficient_attention()
                except Exception as e:
                    print(f"Warning: Could not apply block swap: {e}")

            if model and lora:
                try:
                    # Apply LoRA if supported
                    pass
                except:
                    pass

            if progress_callback:
                progress_callback(0.4, "Loading VAE...")

            # Step 7: Load VAE
            print(f"[DEBUG] Loading VAE: {vae_name}")
            vae_result = self.vae_loader.loadmodel(
                model_name=vae_name,  # Changed from vae_name to model_name
                precision="bf16"
            )
            vae = vae_result[0] if vae_result else None
            print(f"[DEBUG] VAE loaded: {type(vae)}")

            if progress_callback:
                progress_callback(0.45, "Creating image embeddings...")

            # Step 8: Create empty image embeds
            print(f"[DEBUG] Creating embeds with width={width}, height={height}, num_frames={num_frames}")
            embeds_result = self.empty_embeds.process(
                width=width,
                height=height,
                num_frames=num_frames
            )
            image_embeds = embeds_result[0] if embeds_result else None
            print(f"[DEBUG] Image embeds created: {image_embeds.keys() if image_embeds else 'None'}")
            if image_embeds:
                print(f"[DEBUG] Target shape: {image_embeds.get('target_shape')}")
                image_embeds["vae"] = vae

            if progress_callback:
                progress_callback(0.5, "Starting generation...")

            # Step 9: Run sampler
            if seed == -1:
                seed = random.randint(0, 2**63 - 1)

            print(f"[DEBUG] Starting sampler with steps={steps}, cfg={cfg}, seed={seed}")
            print(f"[DEBUG] Image embeds keys: {image_embeds.keys() if image_embeds else 'None'}")
            
            # Fix: text_embeds is a dict, not a tensor
            if text_embeds is not None and isinstance(text_embeds, dict):
                print(f"[DEBUG] Text embeds is dict with keys: {text_embeds.keys()}")
                print(f"[DEBUG] Prompt embeds shape: {text_embeds['prompt_embeds'][0].shape if 'prompt_embeds' in text_embeds else 'None'}")
            else:
                print(f"[DEBUG] Text embeds shape: {text_embeds.shape if text_embeds is not None and hasattr(text_embeds, 'shape') else 'None'}")
            
            print(f"[DEBUG] Model type: {type(model)}")
            
            import time
            start_time = time.time()
            print(f"[DEBUG] Calling sampler.process()...")
            
            samples_result = self.sampler.process(
                model=model,
                image_embeds=image_embeds,
                steps=steps,
                cfg=cfg,
                shift=shift,  # Use the shift parameter from user
                seed=seed,
                scheduler="unipc",  # Using default scheduler
                riflex_freq_index=0,  # Default to 0 (disabled)
                force_offload=True,
                text_embeds=text_embeds,
                cache_args=compile_args,  # Pass compile args to cache_args
                experimental_args=block_swap_args  # Pass block swap args to experimental_args
            )
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Sampler completed in {elapsed:.2f} seconds")
            print(f"[DEBUG] Sampler result type: {type(samples_result)}, length: {len(samples_result) if samples_result else 0}")
            samples = samples_result[0] if samples_result else None
            print(f"[DEBUG] Samples shape: {samples.shape if samples is not None and hasattr(samples, 'shape') else 'None'}")

            if progress_callback:
                progress_callback(0.9, "Decoding video...")

            # Step 10: Decode to video
            print(f"[DEBUG] Starting decoder...")
            video_result = self.decoder.decode(
                vae=vae,
                samples=samples,
                enable_vae_tiling=False,
                tile_x=512,
                tile_y=512,
                tile_stride_x=256,
                tile_stride_y=256
            )
            print(f"[DEBUG] Decoder result type: {type(video_result)}, length: {len(video_result) if video_result else 0}")
            video_frames = video_result[0] if video_result else None
            print(f"[DEBUG] Video frames shape: {video_frames.shape if video_frames is not None and hasattr(video_frames, 'shape') else 'None'}")

            if progress_callback:
                progress_callback(0.95, "Finalizing output...")

            # Convert to numpy array for display
            print(f"[DEBUG] Converting video frames to numpy array...")
            if video_frames is not None and hasattr(video_frames, 'shape'):
                print(f"[DEBUG] Video frames original shape: {video_frames.shape}")
                if hasattr(video_frames, 'cpu'):
                    video_array = video_frames.cpu().numpy()
                else:
                    video_array = video_frames

                # Ensure correct shape [frames, height, width, channels]
                if len(video_array.shape) == 5:
                    # [batch, frames, channels, height, width] -> [frames, height, width, channels]
                    print(f"[DEBUG] Converting from 5D to 4D...")
                    video_array = video_array[0].transpose(0, 2, 3, 1)
                elif len(video_array.shape) == 4:
                    # [frames, channels, height, width] -> [frames, height, width, channels]
                    if video_array.shape[1] == 3:
                        print(f"[DEBUG] Transposing 4D array...")
                        video_array = video_array.transpose(0, 2, 3, 1)

                print(f"[DEBUG] Video array shape after transpose: {video_array.shape}")
                
                # Ensure uint8
                if video_array.max() <= 1.0:
                    print(f"[DEBUG] Converting from float [0,1] to uint8...")
                    video_array = (video_array * 255).astype(np.uint8)
                else:
                    print(f"[DEBUG] Converting to uint8...")
                    video_array = video_array.astype(np.uint8)
                
                print(f"[DEBUG] Final video array shape: {video_array.shape}, dtype: {video_array.dtype}")
            else:
                # Fallback if no video generated
                print(f"[ERROR] No video frames generated! Creating fallback...")
                video_array = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

            # Save video to project output directory (genesis/output)
            import cv2
            from pathlib import Path

            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = str(output_dir / f"wanvideo_{seed}.mp4")
            
            print(f"[DEBUG] Saving video to: {video_path}")
            print(f"[DEBUG] Video array shape: {video_array.shape}, fps: {fps}")

            # Use OpenCV to write video (MP4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"[ERROR] Failed to open video writer!")
                raise Exception(f"Failed to create video file: {video_path}")

            frame_count = 0
            for frame in video_array:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                frame_count += 1

            out.release()
            
            print(f"[DEBUG] Saved {frame_count} frames to video")
            print(f"[DEBUG] Video file size: {Path(video_path).stat().st_size / 1024 / 1024:.2f} MB")

            metadata["output_shape"] = video_array.shape
            metadata["generation_time"] = time.time()
            metadata["output_path"] = video_path

            if progress_callback:
                progress_callback(1.0, "Generation complete!")

            print(f"[SUCCESS] Video generation complete!")
            print(f"[SUCCESS] Output: {video_path}")
            return video_path, video_array, metadata

        except Exception as e:
            import traceback
            error_msg = f"Error during generation: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"\n{'='*60}")
            print(f"ERROR: {error_msg}")
            print(f"{'='*60}")
            print(traceback_str)
            print(f"{'='*60}\n")
            if progress_callback:
                progress_callback(0, error_msg)
            raise gr.Error(error_msg)


def scan_model_files(directory: str, extensions: List[str] = [".safetensors", ".ckpt", ".pt", ".pth"]) -> List[str]:
    """
    Scan directory for model files

    Args:
        directory: Directory to scan
        extensions: File extensions to look for

    Returns:
        List of model file names
    """
    if not os.path.exists(directory):
        return ["No models found (directory not exists)"]

    models = []
    for ext in extensions:
        pattern = os.path.join(directory, f"**/*{ext}")
        files = glob.glob(pattern, recursive=True)
        for file in files:
            # Get relative path from directory
            rel_path = os.path.relpath(file, directory)
            models.append(rel_path)

    if not models:
        models = ["No models found"]

    return sorted(models)


def get_model_directories():
    """Get default model directories"""
    # Get genesis models directory (E:\chai fream\genesis\models\)
    current_dir = Path(__file__).parent  # genesis/apps/
    genesis_dir = current_dir.parent  # genesis/
    models_root = genesis_dir / "models"

    return {
        "models": str(models_root / "unet"),  # Main models in unet folder
        "vae": str(models_root / "vae"),
        "t5": str(models_root / "text_encoders"),  # T5 models in text_encoders folder
        "lora": str(models_root / "loras")
    }


def create_interface():
    """Create the Gradio interface"""

    workflow = WanVideoWorkflow()
    model_dirs = get_model_directories()

    # Scan for available models
    available_models = scan_model_files(model_dirs["models"])
    available_vaes = scan_model_files(model_dirs["vae"])
    available_t5 = scan_model_files(model_dirs["t5"])
    available_loras = scan_model_files(model_dirs["lora"])

    # Check if models exist, provide better messages
    if not available_models or available_models == ["No models found"]:
        available_models = ["Please place model files in 'models/' directory"]
    if not available_vaes or available_vaes == ["No models found"]:
        available_vaes = ["Please place VAE files in 'models/vae/' directory"]
    if not available_t5 or available_t5 == ["No models found"]:
        available_t5 = ["Please place T5 files in 'models/t5/' directory"]
    if not available_loras or available_loras == ["No models found"]:
        available_loras = ["Please place LoRA files in 'models/lora/' directory"]

    def generate_with_progress(*args):
        """Wrapper to handle progress updates"""
        progress = gr.Progress()

        def progress_callback(value, desc):
            progress(value, desc=desc)

        # Unpack arguments
        (positive_prompt, negative_prompt, width, height, num_frames,
         steps, cfg, shift, seed, scheduler, denoise_strength,
         model_name, vae_name, t5_model, base_precision, quantization, attention_mode,
         lora_enabled, lora_name, lora_strength,
         compile_enabled, compile_backend, block_swap_enabled,
         # Output parameters (must match function signature order)
         output_format, fps,
         # Intelligent VRAM Management parameters
         auto_hardware_tuning, vram_threshold_percent, blocks_to_swap,
         enable_cuda_optimization, enable_dram_optimization,
         num_cuda_streams, bandwidth_target,
         offload_txt_emb, offload_img_emb,
         vace_blocks_to_swap, vram_debug_mode) = args

        video_path, video_array, metadata = workflow.generate_video(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            vae_name=vae_name,
            t5_model=t5_model,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            shift=shift,
            seed=seed,
            scheduler=scheduler,
            denoise_strength=denoise_strength,
            base_precision=base_precision,
            quantization=quantization,
            attention_mode=attention_mode,
            lora_enabled=lora_enabled,
            lora_name=lora_name,
            lora_strength=lora_strength,
            compile_enabled=compile_enabled,
            compile_backend=compile_backend,
            block_swap_enabled=block_swap_enabled,
            # Output parameters (must come before default parameters)
            output_format=output_format,
            fps=fps,
            # Intelligent VRAM Management parameters (with defaults)
            auto_hardware_tuning=auto_hardware_tuning,
            vram_threshold_percent=vram_threshold_percent,
            blocks_to_swap=blocks_to_swap,
            enable_cuda_optimization=enable_cuda_optimization,
            enable_dram_optimization=enable_dram_optimization,
            num_cuda_streams=num_cuda_streams,
            bandwidth_target=bandwidth_target,
            offload_txt_emb=offload_txt_emb,
            offload_img_emb=offload_img_emb,
            vace_blocks_to_swap=vace_blocks_to_swap,
            vram_debug_mode=vram_debug_mode,
            progress_callback=progress_callback
        )

        # Create sample frames for preview
        preview_frames = []
        frame_indices = np.linspace(0, len(video_array) - 1, min(8, len(video_array)), dtype=int)
        for idx in frame_indices:
            # Convert to PIL Image for gallery
            from PIL import Image
            frame_pil = Image.fromarray(video_array[idx])
            preview_frames.append(frame_pil)

        metadata_text = json.dumps(metadata, indent=2, default=str)

        return video_path, preview_frames, metadata_text

    # Define scheduler choices (used by both T2V and I2V)
    scheduler_choices = [
        # WanVideo 标准采样器
        "unipc", "unipc/beta",
        "dpm++", "dpm++/beta",
        "dpm++_sde", "dpm++_sde/beta",
        "euler", "euler/beta",
        "deis",
        "lcm", "lcm/beta",
        "res_multistep",
        "flowmatch_causvid",
        "flowmatch_distill",
        "flowmatch_pusa",
        "flowmatch_lowstep_d",
        "flowmatch_sa_ode_stable",
        "sa_ode_stable/lowstep",
        "ode/+",
        "humo_lcm",
        "multitalk",
        "rcm",
        # IChingWuxing 易经五行系列
        "iching/wuxing",
        "iching/wuxing-strong",
        "iching/wuxing-stable",
        "iching/wuxing-smooth",
        "iching/wuxing-clean",
        "iching/wuxing-sharp",
        "iching/wuxing-lowstep",
        # RES4LYF 高级采样器 (Explicit)
        "res_2m",
        "res_2s", "res_3s", "res_5s",
        "deis_2m", "deis_3m", "deis_4m",
        "ralston_2s", "ralston_3s", "ralston_4s",
        "dpmpp_2m", "dpmpp_3m",
        "dpmpp_sde_2s",
        "dpmpp_2s", "dpmpp_3s",
        "midpoint_2s",
        "heun_2s", "heun_3s",
        "houwen-wray_3s",
        "kutta_3s",
        "ssprk3_3s",
        "rk38_4s",
        "rk4_4s",
        "dormand-prince_6s",
        "dormand-prince_13s",
        "bogacki-shampine_7s",
        "ddim",
        # RES4LYF 隐式采样器 (Implicit)
        "gauss-legendre_2s", "gauss-legendre_3s", "gauss-legendre_4s", "gauss-legendre_5s",
        "radau_ia_2s", "radau_ia_3s",
        "radau_iia_2s", "radau_iia_3s",
        "lobatto_iiia_2s", "lobatto_iiia_3s",
        "lobatto_iiib_2s", "lobatto_iiib_3s",
        "lobatto_iiic_2s", "lobatto_iiic_3s",
        "lobatto_iiid_2s", "lobatto_iiid_3s",
        "lobatto_iiistar_2s", "lobatto_iiistar_3s",
        # RES4LYF 对角隐式采样器 (Diagonally Implicit)
        "kraaijevanger_spijker_2s",
        "qin_zhang_2s",
        "pareschi_russo_2s",
        "pareschi_russo_alt_2s",
        "crouzeix_2s", "crouzeix_3s",
        "irk_exp_diag_2s"
    ]
    
    # Create interface with tabs
    with gr.Blocks(title="WanVideo Genesis v3.0", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🎬 WanVideo Genesis v3.0 - Text & Image to Video Generation

            Advanced video generation system powered by Genesis Core and WanVideo models
            
            **✨ New in v3.0:** Image to Video with InfiniteTalk & WanAnimate support!
            """
        )

        with gr.Tabs():
            # Video Generation Tab (统一的视频生成)
            with gr.Tab("📹 视频生成 Video Generation"):
                gr.Markdown("""
                ## 统一视频生成平台
                支持文生视频、图生视频（InfiniteTalk、WanAnimate、Standard I2V）
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Mode selection
                        video_gen_mode = gr.Radio(
                            choices=["文生视频 Text to Video", "图生视频 - InfiniteTalk", "图生视频 - WanAnimate", "图生视频 - Standard I2V"],
                            value="文生视频 Text to Video",
                            label="🎬 生成模式",
                            info="选择视频生成方式"
                        )
                        
                        # Input image (only for image to video modes)
                        with gr.Group(visible=False) as image_input_group:
                            input_image = gr.Image(
                                label="输入图片",
                                type="pil",
                                height=300
                            )
                        
                        # Text inputs
                        positive_prompt = gr.Textbox(
                            label="Positive Prompt",
                            placeholder="Describe your video...",
                            value="镜头跟随穿深蓝色长裙的女人走在教堂走廊",
                            lines=4
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="What to avoid...",
                            value="",
                            lines=2
                        )
                        
                        # Image processing settings (only for image to video modes)
                        with gr.Group(visible=False) as image_processing_group:
                            gr.Markdown("### 📐 图片处理")
                            
                            keep_proportion = gr.Radio(
                                choices=["crop", "pad", "stretch"],
                                value="crop",
                                label="图片适配方式",
                                info="crop: 裁剪(无黑边) | pad: 填充(有黑边) | stretch: 拉伸"
                            )
                            
                            with gr.Row():
                                crop_position = gr.Dropdown(
                                    choices=["center", "top", "bottom", "left", "right"],
                                    value="center",
                                    label="裁剪位置"
                                )
                                
                                upscale_method = gr.Dropdown(
                                    choices=["lanczos", "bicubic", "bilinear", "nearest"],
                                    value="lanczos",
                                    label="缩放算法"
                                )

                        # Basic parameters
                        with gr.Group():
                            gr.Markdown("### 📹 视频参数")
                            with gr.Row():
                                width = gr.Slider(64, 2048, value=1280, step=16, label="Width")
                                height = gr.Slider(64, 2048, value=720, step=16, label="Height")
                            with gr.Row():
                                num_frames = gr.Slider(1, 241, value=61, step=1, label="Frames")
                                fps = gr.Slider(8, 60, value=16, step=1, label="FPS")

                        # Generation parameters
                        with gr.Group():
                            gr.Markdown("### ⚙️ 生成参数")
                            with gr.Row():
                                steps = gr.Slider(1, 100, value=4, step=1, label="Steps")
                                cfg = gr.Slider(0.0, 30.0, value=1.0, step=0.1, label="CFG Scale")
                            with gr.Row():
                                shift = gr.Slider(0.0, 100.0, value=5.0, step=0.1, label="Shift")
                                seed = gr.Number(value=-1, label="Seed (-1 for random)")
                            scheduler = gr.Dropdown(
                                choices=scheduler_choices,
                                value="unipc",
                                label="Scheduler",
                                info="采样器选择"
                            )
                            denoise_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Denoise Strength")
                        
                        # Model configuration
                        with gr.Accordion("🎨 模型配置", open=False):
                            # 检查模型是否有效（不是提示信息）
                            valid_models = [m for m in available_models if m.endswith('.safetensors') or m.endswith('.ckpt')] if available_models else []
                            valid_vaes = [v for v in available_vaes if v.endswith('.safetensors') or v.endswith('.ckpt')] if available_vaes else []
                            valid_t5 = [t for t in available_t5 if t.endswith('.safetensors') or t.endswith('.ckpt')] if available_t5 else []
                            
                            model_name = gr.Dropdown(
                                choices=available_models if available_models else ["No models found"],
                                value=valid_models[0] if valid_models else "Wan2_IceCannon_t2v2.1_nsfw_RCM_Lab_4step.safetensors",
                                label="Diffusion Model",
                                allow_custom_value=True,
                                interactive=True
                            )
                            vae_name = gr.Dropdown(
                                choices=available_vaes if available_vaes else ["No VAE found"],
                                value=valid_vaes[0] if valid_vaes else "Wan2_1_VAE_bf16.safetensors",
                                label="VAE Model",
                                allow_custom_value=True,
                                interactive=True
                            )
                            t5_model = gr.Dropdown(
                                choices=available_t5 if available_t5 else ["No T5 models found"],
                                value=valid_t5[0] if valid_t5 else "models_t5_umt5-xxl-enc-fp8_fully_uncensored.safetensors",
                                label="T5 Text Encoder",
                                allow_custom_value=True,
                                interactive=True
                            )
                        
                        # Advanced settings
                        with gr.Accordion("⚙️ 高级设置", open=False):
                            base_precision = gr.Dropdown(
                                choices=["disabled", "fp32", "bf16", "fp16", "fp16_fast", "fp8_e4m3fn", "fp8_e4m3fn_fast"],
                                value="disabled",
                                label="Base Precision"
                            )
                            quantization = gr.Dropdown(
                                choices=[
                                    "disabled",
                                    "fp8_e4m3fn",
                                    "fp8_e4m3fn_fast",
                                    "fp8_e4m3fn_scaled",
                                    "fp8_e4m3fn_scaled_fast",
                                    "fp8_e5m2",
                                    "fp8_e5m2_fast",
                                    "fp8_e5m2_scaled",
                                    "fp8_e5m2_scaled_fast",
                                    "fp4_experimental",
                                    "fp4_scaled",
                                    "fp4_scaled_fast"
                                ],
                                value="fp8_e4m3fn_fast",
                                label="Quantization",
                                info="fp4需要RTX 5090, fp8_fast需要RTX 4000+系列"
                            )
                            attention_mode = gr.Dropdown(
                                choices=[
                                    "sdpa",
                                    "flash_attn_2",
                                    "flash_attn_3",
                                    "sageattn",
                                    "sageattn_3",
                                    "sageattn_3_fp4",
                                    "sageattn_3_fp8",
                                    "radial_sage_attention"
                                ],
                                value="sageattn",
                                label="Attention Mode",
                                info="sageattn_3_fp4需要RTX 5090, sageattn_3_fp8最快"
                            )
                        
                        # Optimization settings
                        with gr.Accordion("⚡ 性能优化", open=False):
                            gr.Markdown("### Torch Compile")
                            compile_enabled = gr.Checkbox(label="Enable Torch Compile", value=False)
                            compile_backend = gr.Dropdown(
                                choices=["inductor", "eager", "aot_eager", "cudagraphs"],
                                value="inductor",
                                label="Compile Backend"
                            )
                            
                            gr.Markdown("### 智能 VRAM 管理")
                            block_swap_enabled = gr.Checkbox(
                                label="启用智能 VRAM 管理",
                                value=False,
                                info="自动优化 VRAM-DRAM 平衡"
                            )
                            auto_hardware_tuning = gr.Checkbox(
                                label="自动硬件调优",
                                value=True,
                                info="根据 GPU 自动配置（推荐）"
                            )
                            vram_threshold_percent = gr.Slider(
                                30.0, 90.0, value=50.0, step=5.0,
                                label="VRAM 使用阈值 (%)",
                                info="超过此阈值时触发智能迁移"
                            )
                            
                            with gr.Accordion("VRAM 高级参数", open=False):
                                blocks_to_swap = gr.Slider(
                                    0, 40, value=0, step=1,
                                    label="手动分块数",
                                    info="0=自动计算"
                                )
                                enable_cuda_optimization = gr.Checkbox(
                                    label="启用 CUDA 优化",
                                    value=True
                                )
                                enable_dram_optimization = gr.Checkbox(
                                    label="启用 DRAM 优化",
                                    value=True
                                )
                                num_cuda_streams = gr.Slider(
                                    1, 16, value=8, step=1,
                                    label="CUDA 流数量"
                                )
                                bandwidth_target = gr.Slider(
                                    0.1, 1.0, value=0.8, step=0.1,
                                    label="带宽目标"
                                )
                                offload_txt_emb = gr.Checkbox(
                                    label="卸载文本嵌入",
                                    value=False
                                )
                                offload_img_emb = gr.Checkbox(
                                    label="卸载图像嵌入",
                                    value=False
                                )
                                vace_blocks_to_swap = gr.Slider(
                                    0, 15, value=0, step=1,
                                    label="VAE 分块数"
                                )
                                vram_debug_mode = gr.Checkbox(
                                    label="调试模式",
                                    value=False
                                )
                        
                        # LoRA settings
                        with gr.Accordion("🎨 LoRA 设置", open=False):
                            lora_enabled = gr.Checkbox(
                                label="启用 LoRA",
                                value=False,
                                info="使用 LoRA 微调模型"
                            )
                            lora_name = gr.Dropdown(
                                choices=available_loras if available_loras else ["No LoRA found"],
                                value=available_loras[0] if available_loras and available_loras[0] != "No LoRA found" else "Kinesis-T2V-14B_lora_fix.safetensors",
                                label="Select LoRA",
                                allow_custom_value=True,
                                interactive=True
                            )
                            lora_strength = gr.Slider(
                                -2.0, 2.0, value=1.0, step=0.01,
                                label="LoRA Strength",
                                info="LoRA 强度"
                            )
                        
                        # Mode-specific settings
                        # InfiniteTalk settings
                        with gr.Group(visible=False) as infinitetalk_group:
                            gr.Markdown("### 🎙️ InfiniteTalk 设置")
                            audio_file = gr.Audio(
                                label="音频文件 (可选)",
                                type="filepath"
                            )
                            with gr.Accordion("窗口参数", open=False):
                                frame_window_size = gr.Slider(1, 200, value=117, step=4, label="Frame Window Size")
                                motion_frame = gr.Slider(1, 50, value=25, step=1, label="Motion Frame")
                            with gr.Accordion("Wav2Vec 设置", open=False):
                                wav2vec_precision = gr.Radio(
                                    choices=["fp16", "fp32", "bf16"],
                                    value="fp16",
                                    label="模型精度"
                                )
                                wav2vec_device = gr.Radio(
                                    choices=["main_device", "cpu"],
                                    value="main_device",
                                    label="加载设备"
                                )
                        
                        # WanAnimate settings
                        with gr.Group(visible=False) as wananimate_group:
                            gr.Markdown("### 🎭 WanAnimate 设置")
                            pose_images = gr.File(label="姿态图片序列", file_count="multiple")
                            face_images = gr.File(label="面部图片序列", file_count="multiple")
                            with gr.Row():
                                pose_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Pose Strength")
                                face_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Face Strength")
                            colormatch = gr.Dropdown(
                                choices=["none", "mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"],
                                value="mkl",
                                label="Color Match"
                            )

                        generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        # Output
                        video_output = gr.Video(label="Generated Video")
                        with gr.Accordion("Preview Frames", open=False):
                            frames_gallery = gr.Gallery(label="Frame Preview", columns=4, height=200)
                        with gr.Accordion("Generation Metadata", open=False):
                            metadata_output = gr.Code(language="json", label="Metadata")
                
                # Mode switching logic
                def update_video_gen_mode_visibility(mode):
                    """根据模式显示/隐藏相关组件"""
                    is_image_mode = mode != "文生视频 Text to Video"
                    is_infinitetalk = mode == "图生视频 - InfiniteTalk"
                    is_wananimate = mode == "图生视频 - WanAnimate"
                    
                    return (
                        gr.update(visible=is_image_mode),  # image_input_group
                        gr.update(visible=is_image_mode),  # image_processing_group
                        gr.update(visible=is_infinitetalk),  # infinitetalk_group
                        gr.update(visible=is_wananimate)   # wananimate_group
                    )
                
                # Bind mode switching event
                video_gen_mode.change(
                    update_video_gen_mode_visibility,
                    inputs=[video_gen_mode],
                    outputs=[image_input_group, image_processing_group, infinitetalk_group, wananimate_group]
                )
                
                # Unified generation function
                def unified_video_generation(*args):
                    """统一的视频生成函数，根据模式调用不同的生成逻辑"""
                    mode = args[0]  # video_gen_mode
                    
                    print(f"\n{'='*60}")
                    print(f"[统一视频生成] 模式: {mode}")
                    print(f"{'='*60}")
                    
                    # 根据模式调用不同的生成函数
                    if mode == "文生视频 Text to Video":
                        # 文生视频逻辑 - 跳过前面的图生视频参数
                        # args 顺序: mode, input_image, positive_prompt, negative_prompt, 
                        #           keep_proportion, crop_position, upscale_method,
                        #           width, height, num_frames, fps, ...
                        
                        # 提取文生视频需要的参数（跳过 mode 和图片相关参数）
                        _, input_image, pos_prompt, neg_prompt, keep_prop, crop_pos, upscale, \
                        w, h, frames, fps_val, steps_val, cfg_val, shift_val, seed_val, sched, denoise, \
                        model, vae, t5, precision, quant, attn, \
                        compile_en, compile_back, block_swap, auto_tune, vram_thresh, \
                        blocks_swap, cuda_opt, dram_opt, cuda_streams, bandwidth, \
                        txt_emb_off, img_emb_off, vae_blocks, vram_debug, \
                        lora_en, lora_name_val, lora_str, \
                        audio, frame_win, motion, wav_prec, wav_dev, \
                        pose_imgs, face_imgs, pose_str, face_str, color = args
                        
                        print(f"[文生视频] 优化参数:")
                        print(f"  - LoRA: {lora_en} ({lora_name_val if lora_en else 'disabled'})")
                        print(f"  - Compile: {compile_en}")
                        print(f"  - Block Swap: {block_swap}")
                        print(f"  - Auto Tuning: {auto_tune}")
                        
                        # 调用原始文生视频函数
                        return generate_with_progress(
                            pos_prompt, neg_prompt, w, h, frames,
                            steps_val, cfg_val, shift_val, seed_val, sched, denoise,
                            model, vae, t5, precision, quant, attn,
                            lora_en, lora_name_val, lora_str,  # LoRA 参数
                            compile_en, compile_back, block_swap,
                            "mp4", fps_val,  # output_format, fps
                            auto_tune, vram_thresh, blocks_swap,
                            cuda_opt, dram_opt, cuda_streams, bandwidth,
                            txt_emb_off, img_emb_off, vae_blocks, vram_debug
                        )
                    else:
                        # 图生视频逻辑 (InfiniteTalk/WanAnimate/Standard I2V)
                        mode_map = {
                            "图生视频 - InfiniteTalk": "InfiniteTalk",
                            "图生视频 - WanAnimate": "WanAnimate",
                            "图生视频 - Standard I2V": "Standard I2V"
                        }
                        mapped_mode = mode_map.get(mode, "Standard I2V")
                        
                        # 解包所有参数
                        _, input_image, pos_prompt, neg_prompt, keep_prop, crop_pos, upscale, \
                        w, h, frames, fps_val, steps_val, cfg_val, shift_val, seed_val, sched, denoise, \
                        model, vae, t5, precision, quant, attn, \
                        compile_en, compile_back, block_swap, auto_tune, vram_thresh, \
                        blocks_swap, cuda_opt, dram_opt, cuda_streams, bandwidth, \
                        txt_emb_off, img_emb_off, vae_blocks, vram_debug, \
                        lora_en, lora_name_val, lora_str, \
                        audio, frame_win, motion, wav_prec, wav_dev, \
                        pose_imgs, face_imgs, pose_str, face_str, color = args
                        
                        print(f"[图生视频] 映射模式: {mapped_mode}")
                        print(f"[图生视频] 输入图片: {type(input_image)}")
                        print(f"[图生视频] 优化参数:")
                        print(f"  - LoRA: {lora_en} ({lora_name_val if lora_en else 'disabled'})")
                        print(f"  - Compile: {compile_en}")
                        print(f"  - Block Swap: {block_swap}")
                        print(f"  - Auto Tuning: {auto_tune}")
                        
                        # 调用图生视频生成函数
                        progress = gr.Progress()
                        
                        def progress_callback(value, desc):
                            progress(value, desc=desc)
                        
                        try:
                            video_path, video_array, metadata = workflow.generate_image_to_video(
                                input_image=input_image,
                                mode=mapped_mode,
                                positive_prompt=pos_prompt,
                                negative_prompt=neg_prompt,
                                model_name=model,
                                vae_name=vae,
                                t5_model=t5,
                                width=int(w),
                                height=int(h),
                                num_frames=int(frames),
                                steps=int(steps_val),
                                cfg=float(cfg_val),
                                shift=float(shift_val),
                                seed=int(seed_val),
                                scheduler=sched,
                                denoise_strength=float(denoise),
                                base_precision=precision,
                                quantization=quant,
                                attention_mode=attn,
                                audio_file=audio,
                                frame_window_size=int(frame_win) if frame_win else 117,
                                motion_frame=int(motion) if motion else 25,
                                wav2vec_precision=wav_prec,
                                wav2vec_device=wav_dev,
                                keep_proportion=keep_prop,
                                crop_position=crop_pos,
                                upscale_method=upscale,
                                pose_images=pose_imgs,
                                face_images=face_imgs,
                                pose_strength=float(pose_str) if pose_str else 1.0,
                                face_strength=float(face_str) if face_str else 1.0,
                                colormatch=color,
                                fps=int(fps_val),
                                output_format="mp4",
                                # LoRA 参数
                                lora_enabled=lora_en,
                                lora_name=lora_name_val,
                                lora_strength=float(lora_str),
                                # 优化参数
                                compile_enabled=compile_en,
                                compile_backend=compile_back,
                                block_swap_enabled=block_swap,
                                # VRAM 管理参数
                                auto_hardware_tuning=auto_tune,
                                vram_threshold_percent=float(vram_thresh),
                                blocks_to_swap=int(blocks_swap),
                                enable_cuda_optimization=cuda_opt,
                                enable_dram_optimization=dram_opt,
                                num_cuda_streams=int(cuda_streams),
                                bandwidth_target=float(bandwidth),
                                offload_txt_emb=txt_emb_off,
                                offload_img_emb=img_emb_off,
                                vace_blocks_to_swap=int(vae_blocks),
                                vram_debug_mode=vram_debug,
                                progress_callback=progress_callback
                            )
                            
                            # 提取预览帧
                            preview_frames = []
                            if video_array is not None and len(video_array) > 0:
                                frame_indices = [0, len(video_array)//4, len(video_array)//2, 
                                               3*len(video_array)//4, len(video_array)-1]
                                for idx in frame_indices:
                                    if idx < len(video_array):
                                        from PIL import Image
                                        import numpy as np
                                        frame = video_array[idx]
                                        if isinstance(frame, np.ndarray):
                                            if frame.dtype != np.uint8:
                                                frame = (frame * 255).astype(np.uint8)
                                            preview_frames.append(Image.fromarray(frame))
                            
                            return video_path, preview_frames, json.dumps(metadata, indent=2)
                            
                        except Exception as e:
                            import traceback
                            error_msg = f"生成失败: {str(e)}\n\n{traceback.format_exc()}"
                            print(f"[ERROR] {error_msg}")
                            return None, [], json.dumps({"error": error_msg}, indent=2)
                
                # Connect generate button
                generate_btn.click(
                    unified_video_generation,
                    inputs=[
                        video_gen_mode,  # 模式
                        input_image,  # 图片 (可选)
                        positive_prompt, negative_prompt,  # 提示词
                        keep_proportion, crop_position, upscale_method,  # 图片处理
                        width, height, num_frames, fps,  # 视频参数
                        steps, cfg, shift, seed, scheduler, denoise_strength,  # 生成参数
                        model_name, vae_name, t5_model,  # 模型
                        base_precision, quantization, attention_mode,  # 高级设置
                        compile_enabled, compile_backend,  # Torch Compile
                        block_swap_enabled, auto_hardware_tuning, vram_threshold_percent,  # VRAM 管理
                        blocks_to_swap, enable_cuda_optimization, enable_dram_optimization,  # VRAM 高级
                        num_cuda_streams, bandwidth_target, offload_txt_emb, offload_img_emb,
                        vace_blocks_to_swap, vram_debug_mode,
                        lora_enabled, lora_name, lora_strength,  # LoRA 设置
                        audio_file, frame_window_size, motion_frame,  # InfiniteTalk
                        wav2vec_precision, wav2vec_device,
                        pose_images, face_images, pose_strength, face_strength, colormatch  # WanAnimate
                    ],
                    outputs=[video_output, frames_gallery, metadata_output]
                )

            # Image Generation Tab (预留)
            with gr.Tab("🖼️ 图像生成 Image Generation"):
                gr.Markdown("""
                ## 图像生成功能
                
                **即将推出！**
                
                - 文生图 (Text to Image)
                - 图生图 (Image to Image)
                - 多种风格和模型支持
                """)
                
                gr.Markdown("### 🚧 功能开发中...")

            # 旧的图生视频标签页 - 已废弃，隐藏
            with gr.Tab("🖼️ Image to Video (旧版)", visible=False):
                gr.Markdown("""
                ## ⚠️ 此标签页已废弃
                
                请使用新的 "📹 视频生成" 标签页，功能更强大！
                
                ---
                
                ## 图生视频 (Image to Video)
                支持 InfiniteTalk 和 WanAnimate 模型
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Mode selection
                        i2v_mode = gr.Radio(
                            choices=["InfiniteTalk", "WanAnimate", "Standard I2V"],
                            value="Standard I2V",
                            label="模式选择",
                            info="选择图生视频模式"
                        )
                        
                        # Input image
                        input_image = gr.Image(
                            label="输入图片",
                            type="pil",
                            height=400
                        )
                        
                        # Prompt inputs
                        i2v_positive_prompt = gr.Textbox(
                            label="Positive Prompt",
                            placeholder="描述视频内容...",
                            lines=3
                        )
                        i2v_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="避免的内容...",
                            lines=2
                        )
                        
                        # Image processing settings (共通参数)
                        with gr.Group():
                            gr.Markdown("### 📐 图片处理")
                            
                            keep_proportion = gr.Radio(
                                choices=["crop", "pad", "stretch"],
                                value="crop",
                                label="图片适配方式",
                                info="crop: 裁剪多余部分(无黑边) | pad: 添加黑边(完整显示) | stretch: 拉伸(可能变形)"
                            )
                            
                            with gr.Row():
                                crop_position = gr.Dropdown(
                                    choices=["center", "top", "bottom", "left", "right"],
                                    value="center",
                                    label="裁剪位置",
                                    info="仅 crop 模式生效"
                                )
                                
                                upscale_method = gr.Dropdown(
                                    choices=["lanczos", "bicubic", "bilinear", "nearest"],
                                    value="lanczos",
                                    label="缩放算法",
                                    info="lanczos: 最高质量"
                                )
                        
                        # Video settings
                        with gr.Group():
                            gr.Markdown("### 视频参数")
                            with gr.Row():
                                i2v_width = gr.Slider(64, 2048, value=832, step=16, label="Width")
                                i2v_height = gr.Slider(64, 2048, value=480, step=16, label="Height")
                            with gr.Row():
                                i2v_num_frames = gr.Slider(1, 241, value=81, step=4, label="Frames")
                                i2v_fps = gr.Slider(8, 60, value=25, step=1, label="FPS", info="InfiniteTalk推荐25")
                        
                        # Generation parameters
                        with gr.Group():
                            gr.Markdown("### 生成参数")
                            i2v_steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
                            i2v_cfg = gr.Slider(0.0, 30.0, value=6.0, step=0.1, label="CFG Scale")
                            i2v_shift = gr.Slider(0.0, 100.0, value=5.0, step=0.1, label="Shift")
                            i2v_seed = gr.Number(value=-1, label="Seed (-1 for random)")
                            i2v_scheduler = gr.Dropdown(
                                choices=scheduler_choices,
                                value="dpm++_sde",
                                label="Scheduler",
                                info="推荐: unipc或dpm++_sde (不要选multitalk)"
                            )
                            i2v_denoise = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Denoise Strength")
                        
                        # InfiniteTalk specific settings
                        with gr.Group(visible=False) as infinitetalk_settings:
                            gr.Markdown("### InfiniteTalk 设置")
                            audio_file = gr.Audio(
                                label="音频文件 (可选) - 支持 MP3, WAV, FLAC 等格式",
                                type="filepath"
                            )
                            frame_window_size = gr.Slider(
                                1, 200, value=117, step=4,
                                label="Frame Window Size",
                                info="每个窗口的帧数（推荐117）"
                            )
                            motion_frame = gr.Slider(
                                1, 50, value=25, step=1,
                                label="Motion Frame",
                                info="运动帧数/重叠长度（推荐25）"
                            )
                            
                            # Wav2Vec 模型参数
                            with gr.Accordion("🎙️ Wav2Vec 音频模型设置", open=False):
                                wav2vec_precision = gr.Radio(
                                    choices=["fp16", "fp32", "bf16"],
                                    value="fp16",
                                    label="模型精度 (Precision)",
                                    info="fp16: 快速省显存 | fp32: 精度高 | bf16: 平衡"
                                )
                                wav2vec_device = gr.Radio(
                                    choices=["main_device", "offload_device", "cpu"],
                                    value="main_device",
                                    label="加载设备 (Device)",
                                    info="main_device: GPU | offload_device: 自动卸载 | cpu: CPU"
                                )
                        
                        # WanAnimate specific settings
                        with gr.Group(visible=False) as wananimate_settings:
                            gr.Markdown("### WanAnimate 设置")
                            pose_images = gr.Image(
                                label="姿态图片 (可选)",
                                type="pil"
                            )
                            face_images = gr.Image(
                                label="面部图片 (可选)",
                                type="pil"
                            )
                            pose_strength = gr.Slider(0.0, 10.0, value=1.0, step=0.01, label="Pose Strength")
                            face_strength = gr.Slider(0.0, 10.0, value=1.0, step=0.01, label="Face Strength")
                            animate_frame_window = gr.Slider(
                                1, 200, value=77, step=1,
                                label="Frame Window Size"
                            )
                            colormatch = gr.Dropdown(
                                choices=['disabled', 'mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm'],
                                value='mkl',
                                label="Color Match",
                                info="窗口间颜色匹配方法（推荐mkl）"
                            )
                        
                        # Model selection
                        with gr.Group():
                            gr.Markdown("### 模型选择")
                            i2v_model_name = gr.Dropdown(
                                choices=available_models,
                                value=available_models[0] if available_models else None,
                                label="Diffusion Model",
                                allow_custom_value=True
                            )
                            i2v_vae_name = gr.Dropdown(
                                choices=available_vaes,
                                value=available_vaes[0] if available_vaes else None,
                                label="VAE Model",
                                allow_custom_value=True
                            )
                            i2v_t5_model = gr.Dropdown(
                                choices=available_t5,
                                value=available_t5[0] if available_t5 else None,
                                label="T5 Text Encoder",
                                allow_custom_value=True
                            )
                        
                        # Advanced settings
                        with gr.Accordion("高级设置", open=False):
                            i2v_base_precision = gr.Dropdown(
                                choices=["disabled", "fp32", "bf16", "fp16", "fp16_fast", "fp8_e4m3fn", "fp8_e4m3fn_fast"],
                                value="disabled",
                                label="Base Precision"
                            )
                            i2v_quantization = gr.Dropdown(
                                choices=["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp4_scaled"],
                                value="disabled",
                                label="Quantization"
                            )
                            i2v_attention_mode = gr.Dropdown(
                                choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn", "sageattn_3_fp8"],
                                value="sdpa",
                                label="Attention Mode"
                            )
                        
                        i2v_generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                        
                        # Test button to verify click events work
                        test_btn = gr.Button("🧪 Test Click (调试用)", variant="secondary", size="sm", visible=True)
                    
                    with gr.Column(scale=1):
                        # Output
                        i2v_video_output = gr.Video(label="Generated Video")
                        with gr.Accordion("Preview Frames", open=False):
                            i2v_frames_gallery = gr.Gallery(label="Frame Preview", columns=4, height=200)
                        with gr.Accordion("Metadata", open=False):
                            i2v_metadata_output = gr.Code(language="json", label="Metadata")
                
                # Mode change handlers
                def update_i2v_settings(mode):
                    if mode == "InfiniteTalk":
                        return gr.update(visible=True), gr.update(visible=False)
                    elif mode == "WanAnimate":
                        return gr.update(visible=False), gr.update(visible=True)
                    else:
                        return gr.update(visible=False), gr.update(visible=False)
                
                i2v_mode.change(
                    update_i2v_settings,
                    inputs=[i2v_mode],
                    outputs=[infinitetalk_settings, wananimate_settings]
                )
                
                # Image to Video generation function (定义在这里,在标签页内)
                def generate_i2v_with_progress_local(*args):
                    """Wrapper for image to video generation with progress"""
                    print("\n" + "="*60)
                    print("[DEBUG] I2V Generate button clicked!")
                    print(f"[DEBUG] Received {len(args)} arguments")
                    print("="*60)
                    
                    progress = gr.Progress()
                    
                    def progress_callback(value, desc):
                        progress(value, desc=desc)
                    
                    # Unpack arguments
                    try:
                        (input_image, mode, positive_prompt, negative_prompt,
                         model_name, vae_name, t5_model,
                         width, height, num_frames, steps, cfg, shift, seed, scheduler, denoise,
                         base_precision, quantization, attention_mode,
                         audio_file, frame_window_size, motion_frame,
                         wav2vec_precision, wav2vec_device,  # Wav2Vec 参数
                         keep_proportion, crop_position, upscale_method,  # 图片处理参数
                         pose_images, face_images, pose_strength, face_strength, colormatch, fps) = args
                    except ValueError as e:
                        print(f"[ERROR] Failed to unpack arguments: {e}")
                        print(f"[ERROR] Expected 32 args, got {len(args)}")
                        return None, [], json.dumps({"error": f"参数解包失败: {e}"}, indent=2)
                    
                    print(f"[DEBUG] Mode: {mode}")
                    print(f"[DEBUG] Image: {type(input_image)}")
                    print(f"[DEBUG] Frame window size: {frame_window_size}")
                    print(f"[DEBUG] Wav2Vec config: {wav2vec_precision}, {wav2vec_device}")
                    
                    if input_image is None:
                        return None, [], json.dumps({"error": "请上传输入图片"}, indent=2)
                    
                    try:
                        video_path, video_array, metadata = workflow.generate_image_to_video(
                            input_image=input_image,
                            mode=mode,
                            positive_prompt=positive_prompt,
                            negative_prompt=negative_prompt,
                            model_name=model_name,
                            vae_name=vae_name,
                            t5_model=t5_model,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            steps=steps,
                            cfg=cfg,
                            shift=shift,
                            seed=int(seed),
                            scheduler=scheduler,
                            denoise_strength=denoise,
                            base_precision=base_precision,
                            quantization=quantization,
                            attention_mode=attention_mode,
                            audio_file=audio_file,
                            frame_window_size=int(frame_window_size),
                            motion_frame=int(motion_frame),
                            wav2vec_precision=wav2vec_precision,
                            wav2vec_device=wav2vec_device,
                            keep_proportion=keep_proportion,
                            crop_position=crop_position,
                            upscale_method=upscale_method,
                            pose_images=pose_images,
                            face_images=face_images,
                            pose_strength=pose_strength,
                            face_strength=face_strength,
                            colormatch=colormatch,
                            fps=int(fps),
                            progress_callback=progress_callback
                        )
                        
                        # Extract preview frames
                        preview_frames = []
                        frame_indices = [0, len(video_array)//4, len(video_array)//2, 3*len(video_array)//4, len(video_array)-1]
                        
                        for idx in frame_indices:
                            if idx < len(video_array):
                                from PIL import Image
                                frame_pil = Image.fromarray(video_array[idx])
                                preview_frames.append(frame_pil)
                        
                        metadata_text = json.dumps(metadata, indent=2, default=str)
                        
                        return video_path, preview_frames, metadata_text
                        
                    except Exception as e:
                        error_msg = f"生成失败: {str(e)}"
                        print(f"[ERROR] {error_msg}")
                        import traceback
                        traceback.print_exc()
                        return None, [], json.dumps({"error": error_msg}, indent=2)
                
                # 添加简单的测试函数来验证按钮是否工作
                def test_button_click():
                    print("\n" + "="*60)
                    print("🧪 TEST: Button click detected!")
                    print("="*60)
                    return None, [], json.dumps({"status": "Button works!"}, indent=2)
                
                # 先测试按钮是否能响应
                test_btn.click(
                    test_button_click,
                    inputs=[],
                    outputs=[i2v_video_output, i2v_frames_gallery, i2v_metadata_output]
                )
                
                # Connect I2V generate button (在标签页内绑定)
                i2v_generate_btn.click(
                    generate_i2v_with_progress_local,
                    inputs=[
                        input_image, i2v_mode, i2v_positive_prompt, i2v_negative_prompt,
                        i2v_model_name, i2v_vae_name, i2v_t5_model,
                        i2v_width, i2v_height, i2v_num_frames,
                        i2v_steps, i2v_cfg, i2v_shift, i2v_seed, i2v_scheduler, i2v_denoise,
                        i2v_base_precision, i2v_quantization, i2v_attention_mode,
                        audio_file, frame_window_size, motion_frame,
                        wav2vec_precision, wav2vec_device,  # Wav2Vec 参数
                        keep_proportion, crop_position, upscale_method,  # 图片处理参数
                        pose_images, face_images, pose_strength, face_strength, colormatch,
                        i2v_fps
                    ],
                    outputs=[i2v_video_output, i2v_frames_gallery, i2v_metadata_output]
                )

            # Model Settings Tab (已集成到视频生成标签页，保留用于兼容)
            with gr.Tab("Model Settings", visible=False):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Configuration")

                        # Model selection with dropdown
                        model_name = gr.Dropdown(
                            choices=available_models if available_models else ["No models found"],
                            value=available_models[0] if available_models and available_models[0] != "No models found" else "Wan2_IceCannon_t2v2.1_nsfw_RCM_Lab_4step.safetensors",
                            label="Select Model",
                            allow_custom_value=True,
                            interactive=True
                        )

                        # VAE selection with dropdown
                        vae_name = gr.Dropdown(
                            choices=available_vaes if available_vaes else ["No VAE found"],
                            value=available_vaes[0] if available_vaes and available_vaes[0] != "No VAE found" else "Wan2_1_VAE_bf16.safetensors",
                            label="Select VAE",
                            allow_custom_value=True,
                            interactive=True
                        )

                        # T5 selection with dropdown
                        t5_model = gr.Dropdown(
                            choices=available_t5 if available_t5 else ["No T5 models found"],
                            value=available_t5[0] if available_t5 and available_t5[0] != "No T5 models found" else "models_t5_umt5-xxl-enc-fp8_fully_uncensored.safetensors",
                            label="Select T5 Model",
                            allow_custom_value=True,
                            interactive=True
                        )

                        # Refresh button to rescan models
                        refresh_models_btn = gr.Button("🔄 Refresh Model List", size="sm")

                    with gr.Column():
                        gr.Markdown("### Advanced Settings")
                        
                        # Base Precision
                        base_precision = gr.Dropdown(
                            choices=[
                                "disabled",  # Auto-detect
                                "fp32",
                                "bf16", 
                                "fp16",
                                "fp16_fast",
                                "fp8_e4m3fn",
                                "fp8_e4m3fn_fast"
                            ],
                            value="bf16",
                            label="Base Precision",
                            info="基础精度 - disabled为自动检测"
                        )
                        
                        # Quantization
                        quantization = gr.Dropdown(
                            choices=[
                                "disabled",  # Auto-select
                                "fp8_e4m3fn",
                                "fp8_e4m3fn_fast",
                                "fp8_e4m3fn_scaled",
                                "fp8_e4m3fn_scaled_fast",
                                "fp8_e5m2",
                                "fp8_e5m2_fast",
                                "fp8_e5m2_scaled",
                                "fp8_e5m2_scaled_fast",
                                "fp4_experimental",
                                "fp4_scaled",
                                "fp4_scaled_fast"
                            ],
                            value="fp8_e4m3fn_fast",
                            label="Quantization",
                            info="量化方式 - fp4需要RTX 5090, fp8_fast需要RTX 4000+系列"
                        )
                        
                        # Attention Mode
                        attention_mode = gr.Dropdown(
                            choices=[
                                "sdpa",  # PyTorch scaled dot product attention
                                "flash_attn_2",  # Flash Attention 2
                                "flash_attn_3",  # Flash Attention 3
                                "sageattn",  # Sage Attention
                                "sageattn_3",  # Sage Attention 3 (INT8/FP16)
                                "sageattn_3_fp4",  # Sage Attention 3 FP4 (RTX 5090)
                                "sageattn_3_fp8",  # Sage Attention 3 FP8 (最快)
                                "radial_sage_attention"  # Radial Sage Attention
                            ],
                            value="sageattn",
                            label="Attention Mode",
                            info="注意力模式 - sageattn_3_fp4需要RTX 5090, sageattn_3_fp8最快"
                        )
                        output_format = gr.Dropdown(
                            choices=["mp4", "gif", "webm", "frames"],
                            value="mp4",
                            label="Output Format"
                        )

            # LoRA Settings Tab
            with gr.Tab("LoRA"):
                lora_enabled = gr.Checkbox(label="Enable LoRA", value=True)

                # LoRA selection with dropdown
                lora_name = gr.Dropdown(
                    choices=available_loras if available_loras else ["No LoRA found"],
                    value=available_loras[0] if available_loras and available_loras[0] != "No LoRA found" else "Kinesis-T2V-14B_lora_fix.safetensors",
                    label="Select LoRA",
                    allow_custom_value=True,
                    interactive=True
                )

                lora_strength = gr.Slider(-2.0, 2.0, value=1.0, step=0.01, label="LoRA Strength")

            # Optimization Tab (已集成到视频生成标签页，保留用于兼容)
            with gr.Tab("Optimization", visible=False):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Torch Compile")
                        compile_enabled = gr.Checkbox(label="Enable Torch Compile", value=False)
                        compile_backend = gr.Dropdown(
                            choices=["inductor", "eager", "aot_eager", "cudagraphs"],
                            value="inductor",
                            label="Compile Backend"
                        )

                    with gr.Column():
                        gr.Markdown("### 智能 VRAM 管理 (Intelligent VRAM Management)")
                        gr.Markdown("*基于 IntelligentVRAMNode 的智能显存管理系统*")
                        
                        # 主开关
                        block_swap_enabled = gr.Checkbox(
                            label="启用智能 VRAM 管理 (Enable Intelligent VRAM)",
                            value=False,
                            info="自动优化 VRAM-DRAM 平衡，防止显存溢出"
                        )
                        
                        # 自动调优
                        auto_hardware_tuning = gr.Checkbox(
                            label="自动硬件调优 (Auto Hardware Tuning)",
                            value=True,
                            info="根据 GPU 型号自动配置最优参数（推荐）"
                        )
                        
                        # VRAM 阈值
                        vram_threshold_percent = gr.Slider(
                            30.0, 90.0, value=50.0, step=5.0,
                            label="VRAM 使用阈值 (VRAM Threshold %)",
                            info="超过此阈值时触发智能迁移"
                        )
                        
                        with gr.Accordion("高级参数 (Advanced Parameters)", open=False):
                            # 手动分块数
                            blocks_to_swap = gr.Slider(
                                0, 40, value=0, step=1,
                                label="手动分块数 (Manual Blocks to Swap)",
                                info="0=自动计算，>0=手动设置（关闭自动调优时生效）"
                            )
                            
                            # CUDA 优化
                            enable_cuda_optimization = gr.Checkbox(
                                label="启用 CUDA 优化 (Enable CUDA Optimization)",
                                value=True,
                                info="多流并行传输，提升迁移效率"
                            )
                            
                            # DRAM 优化
                            enable_dram_optimization = gr.Checkbox(
                                label="启用 DRAM 优化 (Enable DRAM Optimization)",
                                value=True,
                                info="智能 DRAM 缓冲，防止溢出"
                            )
                            
                            # CUDA 流数量
                            num_cuda_streams = gr.Slider(
                                1, 16, value=8, step=1,
                                label="CUDA 流数量 (CUDA Streams)",
                                info="影响并行传输性能（自动调优时自动设置）"
                            )
                            
                            # 带宽目标
                            bandwidth_target = gr.Slider(
                                0.1, 1.0, value=0.8, step=0.1,
                                label="带宽目标 (Bandwidth Target)",
                                info="内存使用目标比例"
                            )
                            
                            # 嵌入卸载
                            offload_txt_emb = gr.Checkbox(
                                label="卸载文本嵌入 (Offload Text Embeddings)",
                                value=False,
                                info="将文本嵌入移至 DRAM"
                            )
                            
                            offload_img_emb = gr.Checkbox(
                                label="卸载图像嵌入 (Offload Image Embeddings)",
                                value=False,
                                info="将图像嵌入移至 DRAM"
                            )
                            
                            # VAE 分块
                            vace_blocks_to_swap = gr.Slider(
                                0, 15, value=0, step=1,
                                label="VAE 分块数 (VAE Blocks to Swap)",
                                info="VAE 的分块交换数量"
                            )
                            
                            # 调试模式
                            vram_debug_mode = gr.Checkbox(
                                label="调试模式 (Debug Mode)",
                                value=False,
                                info="输出详细的 VRAM 管理日志"
                            )

            # Presets Tab
            with gr.Tab("Presets"):
                gr.Markdown(
                    """
                    ### Quick Presets

                    Select a preset configuration for common use cases:
                    """
                )

                preset_buttons = gr.Radio(
                    choices=[
                        "Fast Preview (4 steps, low quality)",
                        "Standard (30 steps, balanced)",
                        "High Quality (50 steps, best quality)",
                        "Memory Optimized (Block swap enabled)",
                        "Speed Optimized (Compile enabled)"
                    ],
                    label="Preset Configuration"
                )

                def apply_preset(preset_name):
                    if "Fast Preview" in preset_name:
                        return 4, 1.0, 5.0, "sa_ode_stable/lowstep", False, False, True, 50.0
                    elif "Standard" in preset_name:
                        return 30, 6.0, 5.0, "unipc", False, False, True, 50.0
                    elif "High Quality" in preset_name:
                        return 50, 8.0, 5.0, "ddim", False, False, True, 50.0
                    elif "Memory Optimized" in preset_name:
                        return 30, 6.0, 5.0, "unipc", False, True, True, 40.0  # 降低阈值
                    elif "Speed Optimized" in preset_name:
                        return 30, 6.0, 5.0, "unipc", True, False, True, 60.0  # 提高阈值
                    else:
                        return 30, 6.0, 5.0, "unipc", False, False, True, 50.0

                preset_buttons.change(
                    apply_preset,
                    inputs=[preset_buttons],
                    outputs=[steps, cfg, shift, scheduler, compile_enabled, block_swap_enabled, 
                            auto_hardware_tuning, vram_threshold_percent]
                )

            # Scheduler Guide Tab
            with gr.Tab("📖 Scheduler Guide"):
                gr.Markdown("""
                # 采样器使用指南 (Scheduler Guide)
                
                ## 🎯 推荐采样器
                
                ### 快速生成 (4-8步)
                - **unipc** - 通用快速采样器,平衡质量和速度 ⭐推荐
                - **dpm++** - DPM++ 采样器,质量较好
                - **euler** - 简单高效的欧拉采样器
                - **sa_ode_stable/lowstep** - 低步数优化,4步即可
                - **ode/+** - 优化的ODE采样器,8步最佳
                - **rcm** - RCM快速采样,4步专用
                
                ### 标准生成 (20-30步)
                - **unipc** - 标准推荐 ⭐
                - **dpm++_sde** - 更好的细节,但速度稍慢
                - **euler/beta** - Beta版本,更稳定
                - **res_multistep** - 多步残差采样
                
                ### 高质量生成 (50+步)
                - **dpm++/beta** - 高质量DPM++
                - **deis** - DEIS采样器,高质量
                - **flowmatch_sa_ode_stable** - 稳定的流匹配
                
                ---
                
                ## 🌟 易经五行采样器 (IChingWuxing)
                
                基于中国传统易经五行理论的创新采样器,不同模式适合不同风格:
                
                - **iching/wuxing** - 标准五行模式,平衡各元素
                - **iching/wuxing-strong** - 强化模式,增强对比度和细节
                - **iching/wuxing-stable** - 稳定模式,减少噪声,画面更干净
                - **iching/wuxing-smooth** - 平滑模式,过渡更自然
                - **iching/wuxing-clean** - 清晰模式,画面更锐利
                - **iching/wuxing-sharp** - 锐化模式,细节更突出
                - **iching/wuxing-lowstep** - 低步数优化,适合快速生成
                
                **使用建议:**
                - 人物视频: `iching/wuxing-stable` 或 `iching/wuxing-smooth`
                - 风景视频: `iching/wuxing-strong` 或 `iching/wuxing-sharp`
                - 快速预览: `iching/wuxing-lowstep` (4-8步)
                
                ---
                
                ## 🚀 特殊用途采样器
                
                ### LCM系列 (快速生成)
                - **lcm** - Latent Consistency Model,4-8步即可
                - **lcm/beta** - Beta版本
                - **humo_lcm** - HuMo LCM优化版本
                
                ### FlowMatch系列 (流匹配)
                - **flowmatch_pusa** - Pusa流匹配
                - **flowmatch_causvid** - CausVid专用
                - **flowmatch_distill** - 蒸馏版本,4步专用
                - **flowmatch_lowstep_d** - 低步数优化
                
                ### 其他
                - **multitalk** - MultiTalk专用采样器
                
                ---
                
                ## ⚙️ 参数建议
                
                | 采样器 | 推荐步数 | 推荐CFG | 推荐Shift |
                |--------|---------|---------|-----------|
                | unipc | 20-30 | 6.0-8.0 | 5.0 |
                | iching/wuxing | 20-40 | 5.0-7.0 | 5.0-7.0 |
                | dpm++ | 25-35 | 7.0-9.0 | 5.0 |
                | euler | 30-50 | 6.0-8.0 | 5.0 |
                | lcm | 4-8 | 1.0-2.0 | 3.0-5.0 |
                | rcm | 4 | 1.0 | 5.0 |
                | ode/+ | 8 | 5.0-7.0 | 5.0 |
                
                ---
                
                ---
                
                ## 🔬 RES4LYF 高级采样器 (Runge-Kutta方法)
                
                基于数值分析的高精度采样器,提供卓越的图像质量:
                
                ### Explicit 显式采样器 (推荐)
                
                **RES 系列** (最佳质量):
                - **res_2m** - 2阶多步,速度快 ⭐推荐
                - **res_2s** - 2阶子步,质量好
                - **res_3s** - 3阶子步,质量更好
                - **res_5s** - 5阶子步,最高质量 ⭐⭐⭐
                
                **DEIS 系列** (快速高质量):
                - **deis_2m/3m/4m** - 多步方法,速度快
                
                **DPM++ 系列** (兼容性好):
                - **dpmpp_2m/3m** - 多步版本
                - **dpmpp_2s/3s** - 子步版本
                - **dpmpp_sde_2s** - SDE版本,细节更好
                
                **经典 RK 方法**:
                - **rk4_4s** - 经典4阶RK,稳定可靠
                - **dormand-prince_6s** - 6阶高精度
                - **dormand-prince_13s** - 13阶超高精度
                
                ### Implicit 隐式采样器 (超高质量)
                
                ⚠️ **速度慢,但质量极高**
                
                - **gauss-legendre_2s/3s/4s/5s** - Gauss-Legendre方法
                - **radau_ia/iia_2s/3s** - Radau方法
                - **lobatto_iii(a/b/c/d/star)_2s/3s** - Lobatto方法
                
                **使用建议:**
                - 适合追求极致质量的场景
                - 通常只需20步即可达到其他采样器50+步的质量
                - 配合 `res_5s` 作为预测器效果最佳
                
                ### Diagonally Implicit 对角隐式采样器
                
                - **crouzeix_2s/3s** - Crouzeix方法
                - **pareschi_russo_2s** - Pareschi-Russo方法
                - **irk_exp_diag_2s** - 指数积分器
                
                ### 📊 RES4LYF 性能对比
                
                | 采样器 | 速度 | 质量 | 步数建议 |
                |--------|------|------|----------|
                | res_2m | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 20-30 |
                | res_2s | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 15-25 |
                | res_5s | ⭐⭐ | ⭐⭐⭐⭐⭐ | 10-20 |
                | gauss-legendre_5s | ⭐ | ⭐⭐⭐⭐⭐ | 10-15 |
                | dpmpp_2m | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 20-30 |
                
                **注意**: 
                - 名称中的数字表示阶数
                - "m" = 多步(multistep),速度快
                - "s" = 子步(substep),质量高但速度慢
                - 子步数越多,每步的模型调用次数越多
                
                ---
                
                ## 💡 使用技巧
                
                1. **快速测试**: 使用 `unipc` + 4步 + CFG 1.0
                2. **标准质量**: 使用 `unipc` 或 `res_2m` + 20-30步 + CFG 6.0
                3. **最佳质量**: 使用 `res_5s` + 15步 + CFG 8.0
                4. **极致质量**: 使用 `gauss-legendre_5s` + 10步 + CFG 8.0
                5. **易经风格**: 根据内容选择对应的 `iching/wuxing-*` 模式
                6. **超快生成**: 使用 `rcm` + 4步 + CFG 1.0 (需要RCM模型)
                7. **RES4LYF快速**: 使用 `res_2m` + 20步 + CFG 6.0
                8. **RES4LYF高质量**: 使用 `res_5s` + 15步 + CFG 7.0
                
                **注意**: 不同采样器对模型和参数的要求不同,建议先用默认参数测试!
                """)

            # Quantization & Attention Guide Tab
            with gr.Tab("⚙️ Quantization & Attention"):
                gr.Markdown("""
                # 量化与注意力模式指南
                
                ## 🎯 基础精度 (Base Precision)
                
                控制模型基础层的数据类型:
                
                - **disabled** - 自动检测(推荐) ⭐
                - **fp32** - 32位浮点,最高精度,显存占用最大
                - **bf16** - BFloat16,平衡精度和速度 ⭐推荐
                - **fp16** - 16位浮点,速度快,显存占用小
                - **fp16_fast** - FP16快速模式,启用 allow_fp16_accumulation
                - **fp8_e4m3fn** - FP8基础精度,自动启用FP8量化
                - **fp8_e4m3fn_fast** - FP8快速模式(需要RTX 4000+系列)
                
                ---
                
                ## 📦 量化方式 (Quantization)
                
                ### FP8 系列 (推荐)
                
                **E4M3FN 格式** (更高精度):
                - **fp8_e4m3fn** - 标准FP8量化 ⭐
                - **fp8_e4m3fn_fast** - 快速FP8(需要RTX 4000+) ⭐⭐
                - **fp8_e4m3fn_scaled** - 缩放FP8,需要专门的scaled权重
                - **fp8_e4m3fn_scaled_fast** - 快速缩放FP8
                
                **E5M2 格式** (更大动态范围):
                - **fp8_e5m2** - 标准E5M2格式
                - **fp8_e5m2_fast** - 快速E5M2(需要RTX 4000+)
                - **fp8_e5m2_scaled** - 缩放E5M2
                - **fp8_e5m2_scaled_fast** - 快速缩放E5M2
                
                ### FP4 系列 (极致压缩)
                
                ⚠️ **需要 RTX 5090 或更高显卡**
                
                - **fp4_experimental** - FP4实验模式,使用FP8权重+FP4注意力
                - **fp4_scaled** - FP4缩放模式,需要scaled权重
                - **fp4_scaled_fast** - 快速FP4缩放模式
                
                **使用建议:**
                - 必须配合 `sageattn_3_fp4` 注意力模式
                - 显存占用最小,速度最快
                - 仅RTX 5090支持(1038 TOPS算力)
                
                ---
                
                ## 🧠 注意力模式 (Attention Mode)
                
                ### 标准模式
                
                - **sdpa** - PyTorch标准注意力,兼容性最好 ⭐
                - **flash_attn_2** - Flash Attention 2,速度快
                - **flash_attn_3** - Flash Attention 3,最新版本
                
                ### SageAttention 系列 (推荐)
                
                - **sageattn** - Sage Attention标准版 ⭐推荐
                - **sageattn_3** - Sage Attention 3 (INT8/FP16混合)
                - **sageattn_3_fp8** - Sage Attention 3 FP8模式,最快速度 ⭐⭐
                - **sageattn_3_fp4** - Sage Attention 3 FP4模式(仅RTX 5090)
                - **radial_sage_attention** - 径向Sage注意力
                
                **SageAttention 优势:**
                - 速度更快
                - 显存占用更小
                - 支持FP8/FP4加速
                
                ---
                
                ## 🎮 显卡推荐配置
                
                ### RTX 5090 (1038 TOPS)
                ```
                Base Precision: fp8_e4m3fn_fast
                Quantization: fp4_scaled_fast
                Attention: sageattn_3_fp4
                ```
                
                ### RTX 4090 / 4080 (8.9+ 算力)
                ```
                Base Precision: bf16
                Quantization: fp8_e4m3fn_fast
                Attention: sageattn_3_fp8
                ```
                
                ### RTX 4070 / 4060
                ```
                Base Precision: bf16
                Quantization: fp8_e4m3fn
                Attention: sageattn
                ```
                
                ### RTX 3090 / 3080 (算力 < 8.9)
                ```
                Base Precision: bf16
                Quantization: fp8_e5m2
                Attention: sageattn
                ```
                
                ### RTX 3070 / 3060
                ```
                Base Precision: fp16
                Quantization: disabled
                Attention: sdpa
                ```
                
                ---
                
                ## ⚠️ 重要注意事项
                
                1. **Fast 模式要求**
                   - `fp8_xxx_fast` 需要 CUDA 算力 >= 8.9 (RTX 4000系列+)
                   - RTX 3000系列及以下请使用非fast版本
                
                2. **Scaled 模式**
                   - 需要使用专门的 scaled 权重文件
                   - 不能与普通权重混用
                
                3. **FP4 模式**
                   - 仅 RTX 5090 支持
                   - 必须配合 `sageattn_3_fp4` 注意力
                   - 需要 FP4 scaled 权重
                
                4. **兼容性**
                   - 不确定时使用 `disabled` 自动检测
                   - `bf16` + `fp8_e4m3fn` + `sageattn` 是最安全的组合
                
                ---
                
                ## 💡 性能对比
                
                | 配置 | 速度 | 显存 | 质量 | 兼容性 |
                |------|------|------|------|--------|
                | FP32 + SDPA | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
                | BF16 + SDPA | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
                | BF16 + FP8 + SageAttn | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
                | BF16 + FP8_Fast + SageAttn3_FP8 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
                | FP8_Fast + FP4 + SageAttn3_FP4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
                
                **推荐**: BF16 + FP8 + SageAttn 是最佳平衡点! ⭐⭐⭐⭐⭐
                """)

        # Refresh models function
        def refresh_all_models():
            """Refresh all model lists"""
            model_dirs = get_model_directories()

            # Rescan models
            new_models = scan_model_files(model_dirs["models"])
            new_vaes = scan_model_files(model_dirs["vae"])
            new_t5 = scan_model_files(model_dirs["t5"])
            new_loras = scan_model_files(model_dirs["lora"])

            # Check if models exist, provide better messages
            if not new_models or new_models == ["No models found"]:
                new_models = ["Please place model files in 'models/' directory"]
            if not new_vaes or new_vaes == ["No models found"]:
                new_vaes = ["Please place VAE files in 'models/vae/' directory"]
            if not new_t5 or new_t5 == ["No models found"]:
                new_t5 = ["Please place T5 files in 'models/t5/' directory"]
            if not new_loras or new_loras == ["No models found"]:
                new_loras = ["Please place LoRA files in 'models/lora/' directory"]

            return (
                gr.Dropdown(choices=new_models if new_models else ["No models found"]),
                gr.Dropdown(choices=new_vaes if new_vaes else ["No VAE found"]),
                gr.Dropdown(choices=new_t5 if new_t5 else ["No T5 models found"]),
                gr.Dropdown(choices=new_loras if new_loras else ["No LoRA found"])
            )

        # Connect refresh button
        refresh_models_btn.click(
            refresh_all_models,
            outputs=[model_name, vae_name, t5_model, lora_name]
        )

        # Connect generate button
        generate_btn.click(
            generate_with_progress,
            inputs=[
                positive_prompt, negative_prompt, width, height, num_frames,
                steps, cfg, shift, seed, scheduler, denoise_strength,
                model_name, vae_name, t5_model, base_precision, quantization, attention_mode,
                lora_enabled, lora_name, lora_strength,
                compile_enabled, compile_backend, block_swap_enabled,
                # Output parameters (must come before default parameters)
                output_format, fps,
                # Intelligent VRAM Management parameters (with defaults)
                auto_hardware_tuning, vram_threshold_percent, blocks_to_swap,
                enable_cuda_optimization, enable_dram_optimization,
                num_cuda_streams, bandwidth_target,
                offload_txt_emb, offload_img_emb,
                vace_blocks_to_swap, vram_debug_mode
            ],
            outputs=[video_output, frames_gallery, metadata_output]
        )

        # Examples
        gr.Examples(
            examples=[
                ["A majestic eagle soaring through mountain peaks at sunset", "", 1280, 720, 61],
                ["镜头跟随穿深蓝色长裙的女人走在教堂走廊", "", 1280, 720, 61],
                ["Underwater coral reef with colorful fish swimming", "", 1280, 720, 61],
                ["Time-lapse of flowers blooming in a garden", "", 1280, 720, 61],
            ],
            inputs=[positive_prompt, negative_prompt, width, height, num_frames],
            label="Example Prompts"
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",  # Localhost only for stability
        server_port=7860,
        share=False,
        inbrowser=True,
        quiet=False
    )
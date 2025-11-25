"""
配置管理器 - 读写 INI 配置文件
"""
import os
import json
import configparser
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """视频生成参数配置管理器"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # 默认配置文件路径
            genesis_dir = Path(__file__).parent.parent
            config_path = genesis_dir / "config" / "video_params.ini"
        
        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()
        
        # 确保配置目录存在
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.load()
    
    def load(self):
        """加载配置文件"""
        if self.config_path.exists():
            self.config.read(self.config_path, encoding='utf-8')
            print(f"[ConfigManager] Loaded config from: {self.config_path}")
        else:
            print(f"[ConfigManager] Config file not found, will create: {self.config_path}")
            self._create_default_config()
    
    def save(self):
        """保存配置文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            self.config.write(f)
        print(f"[ConfigManager] Saved config to: {self.config_path}")
    
    def _create_default_config(self):
        """创建默认配置"""
        self.config['VideoGeneration'] = {
            'model_name': 'Wan2_IceCannon_t2v2.1_nsfw_RCM_Lab_4step.safetensors',
            'vae_name': 'Wan2_1_VAE_bf16.safetensors',
            't5_model': 'umt5-xxl-enc-fp8_e4m3fn.safetensors',
            'width': '512',
            'height': '512',
            'frames': '16',
            'fps': '8',
            'steps': '20',
            'cfg_scale': '7.5',
            'scheduler': 'unipc',
            'seed': '-1',
            'base_precision': 'bf16',
            'quantization': 'fp8_e4m3fn_fast_scaled',
            'attention_mode': 'auto',
            'shift': '1.0',
            'denoise_strength': '1.0',
            'output_format': 'mp4',
            'motion_strength': '0.5',
        }
        
        self.config['LastUsed'] = {
            'prompt': '',
            'negative_prompt': '',
            'last_width': '512',
            'last_height': '512',
            'last_frames': '16',
            'last_fps': '8',
            'last_steps': '20',
            'last_cfg_scale': '7.5',
            'last_scheduler': 'unipc',
            'last_seed': '-1',
        }
        
        self.save()
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        if 'VideoGeneration' not in self.config:
            self._create_default_config()
        
        section = self.config['VideoGeneration']
        return {
            'model_name': section.get('model_name', 'Wan2_IceCannon_t2v2.1_nsfw_RCM_Lab_4step.safetensors'),
            'vae_name': section.get('vae_name', 'Wan2_1_VAE_bf16.safetensors'),
            't5_model': section.get('t5_model', 'umt5-xxl-enc-fp8_e4m3fn.safetensors'),
            'width': section.getint('width', 512),
            'height': section.getint('height', 512),
            'frames': section.getint('frames', 16),
            'fps': section.getint('fps', 8),
            'steps': section.getint('steps', 20),
            'cfg_scale': section.getfloat('cfg_scale', 7.5),
            'scheduler': section.get('scheduler', 'unipc'),
            'seed': section.getint('seed', -1),
            'base_precision': section.get('base_precision', 'bf16'),
            'quantization': section.get('quantization', 'fp8_e4m3fn_fast_scaled'),
            'attention_mode': section.get('attention_mode', 'auto'),
            'shift': section.getfloat('shift', 1.0),
            'denoise_strength': section.getfloat('denoise_strength', 1.0),
            'output_format': section.get('output_format', 'mp4'),
            'motion_strength': section.getfloat('motion_strength', 0.5),
        }
    
    def get_last_used_params(self) -> Dict[str, Any]:
        """获取上次使用的参数"""
        if 'LastUsed' not in self.config:
            return {}
        
        section = self.config['LastUsed']
        
        # 解析 LoRA 列表
        loras = []
        loras_json = section.get('loras', '[]')
        try:
            loras = json.loads(loras_json)
        except:
            loras = []
        
        return {
            'prompt': section.get('prompt', ''),
            'negative_prompt': section.get('negative_prompt', ''),
            'model_id': section.get('model_id', ''),
            'width': section.getint('last_width', 512),
            'height': section.getint('last_height', 512),
            'frames': section.getint('last_frames', 16),
            'fps': section.getint('last_fps', 8),
            'steps': section.getint('last_steps', 20),
            'cfg_scale': section.getfloat('last_cfg_scale', 7.5),
            'scheduler': section.get('last_scheduler', 'unipc'),
            'seed': section.getint('last_seed', -1),
            'shift': section.getfloat('last_shift', 1.0),
            'loras': loras,
            'last_video_url': section.get('last_video_url', ''),
            'last_video_info': section.get('last_video_info', ''),
        }
    
    def save_last_used_params(self, params: Dict[str, Any]):
        """保存上次使用的参数"""
        if 'LastUsed' not in self.config:
            self.config['LastUsed'] = {}
        
        section = self.config['LastUsed']
        
        # 保存提示词
        if 'prompt' in params:
            section['prompt'] = str(params['prompt'])
        if 'negative_prompt' in params:
            section['negative_prompt'] = str(params['negative_prompt'])
        
        # 保存模型选择
        if 'model_id' in params:
            section['model_id'] = str(params['model_id'])
        
        # 保存参数
        if 'width' in params:
            section['last_width'] = str(params['width'])
        if 'height' in params:
            section['last_height'] = str(params['height'])
        if 'frames' in params:
            section['last_frames'] = str(params['frames'])
        if 'fps' in params:
            section['last_fps'] = str(params['fps'])
        if 'steps' in params:
            section['last_steps'] = str(params['steps'])
        if 'cfg_scale' in params:
            section['last_cfg_scale'] = str(params['cfg_scale'])
        if 'scheduler' in params:
            section['last_scheduler'] = str(params['scheduler'])
        if 'seed' in params:
            section['last_seed'] = str(params['seed'])
        if 'shift' in params:
            section['last_shift'] = str(params['shift'])
        
        # 保存 LoRA 列表（序列化为 JSON）
        if 'loras' in params:
            try:
                section['loras'] = json.dumps(params['loras'], ensure_ascii=False)
            except:
                section['loras'] = '[]'
        
        # 保存上次生成的视频 URL
        if 'last_video_url' in params:
            section['last_video_url'] = str(params['last_video_url'])
        if 'last_video_info' in params:
            section['last_video_info'] = str(params['last_video_info'])
        
        self.save()
        print(f"[ConfigManager] Saved last used params")
    
    def update_default_param(self, key: str, value: Any):
        """更新默认参数"""
        if 'VideoGeneration' not in self.config:
            self._create_default_config()
        
        self.config['VideoGeneration'][key] = str(value)
        self.save()
        print(f"[ConfigManager] Updated default param: {key} = {value}")
    
    def get_all_config(self) -> Dict[str, Dict[str, Any]]:
        """获取所有配置"""
        return {
            'defaults': self.get_default_params(),
            'last_used': self.get_last_used_params(),
        }


# 全局配置管理器实例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

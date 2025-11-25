import requests
import time
import json
import io
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from .remote_global import get_remote_engine

class RemoteComfyUIClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
    
    def queue_prompt(self, workflow: dict) -> str:
        url = f"{self.server_url}/prompt"
        response = self.session.post(url, json={"prompt": workflow})
        response.raise_for_status()
        return response.json()['prompt_id']
    
    def get_history(self, prompt_id: str) -> Optional[dict]:
        url = f"{self.server_url}/history/{prompt_id}"
        response = self.session.get(url)
        response.raise_for_status()
        history = response.json()
        return history.get(prompt_id)
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        url = f"{self.server_url}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 3600, callback=None) -> dict:
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if history and history.get('status', {}).get('completed', False):
                return history
            if callback:
                callback(prompt_id, int(time.time() - start_time))
            time.sleep(2)
        raise TimeoutError(f"Task {prompt_id} timed out after {timeout}s")


class RemoteWanVideoSampler(ComfyNodeABC):
    """
    Remote ComfyUI WanVideo Sampler - Execute WanVideo on remote GPU
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "remote_server": (IO.STRING, {
                    "default": "http://127.0.0.1:8188",
                    "multiline": False,
                    "tooltip": "Remote ComfyUI server URL (e.g., http://192.168.1.100:8188 or https://app.wwwan21.xyz/comfyui)"
                }),
                "model_name": (IO.STRING, {
                    "default": "wan_video_14b.safetensors",
                    "tooltip": "Model name on remote server"
                }),
                "positive_prompt": (IO.STRING, {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Positive prompt"
                }),
                "negative_prompt": (IO.STRING, {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Negative prompt"
                }),
                "width": (IO.INT, {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": (IO.INT, {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "num_frames": (IO.INT, {"default": 81, "min": 1, "max": 500, "step": 4}),
                "steps": (IO.INT, {"default": 30, "min": 1, "max": 100}),
                "cfg": (IO.FLOAT, {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "timeout": (IO.INT, {
                    "default": 3600,
                    "min": 60,
                    "max": 7200,
                    "tooltip": "Task timeout in seconds"
                }),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute_remote"
    CATEGORY = "WanVideoWrapper/Remote"
    DESCRIPTION = "Execute WanVideo generation on remote ComfyUI server"
    
    async def execute_remote(
        self,
        remote_server: str,
        model_name: str,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        seed: int,
        timeout: int = 3600,
        **kwargs
    ):
        server = self._resolve_server(remote_server)
        client = RemoteComfyUIClient(server)
        
        # Build remote workflow
        workflow = self._build_workflow(
            model_name=model_name,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed
        )
        
        # Submit to remote
        print(f"[RemoteWanVideo] Submitting to {server}")
        prompt_id = client.queue_prompt(workflow)
        print(f"[RemoteWanVideo] Task ID: {prompt_id}")
        
        # Wait for completion
        def progress_callback(pid, elapsed):
            print(f"[RemoteWanVideo] Waiting... {elapsed}s")
        
        history = client.wait_for_completion(prompt_id, timeout, progress_callback)
        
        # Extract results
        images = self._extract_images(client, history)
        return (images,)
    
    def _resolve_server(self, remote_server: str) -> str:
        s = (remote_server or "").strip()
        if s:
            return s
        eng = get_remote_engine()
        if eng.get("enabled") and eng.get("base"):
            return str(eng["base"]).strip()
        raise ValueError("Remote server not specified and global remote engine is not enabled.")

    def _build_workflow(self, **params) -> dict:
        """Build WanVideo workflow for remote execution"""
        return {
            "1": {
                "class_type": "WanVideoModelLoader",
                "inputs": {
                    "model_name": params['model_name'],
                    "precision": "bf16"
                }
            },
            "2": {
                "class_type": "WanVideoTextEncodeCached",
                "inputs": {
                    "model_name": "umt5-xxl-enc",
                    "precision": "bf16",
                    "positive_prompt": params['positive_prompt'],
                    "negative_prompt": params['negative_prompt']
                }
            },
            "3": {
                "class_type": "WanVideoImageEmbeds",
                "inputs": {
                    "width": params['width'],
                    "height": params['height'],
                    "num_frames": params['num_frames']
                }
            },
            "4": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": ["1", 0],
                    "text_embeds": ["2", 0],
                    "image_embeds": ["3", 0],
                    "steps": params['steps'],
                    "cfg": params['cfg'],
                    "seed": params['seed'],
                    "shift": 5.0,
                    "scheduler": "unipc",
                    "riflex_freq_index": 0,
                    "force_offload": True
                }
            },
            "5": {
                "class_type": "WanVideoVAELoader",
                "inputs": {
                    "model_name": "wan_video_vae.safetensors",
                    "precision": "bf16"
                }
            },
            "6": {
                "class_type": "WanVideoVAEDecode",
                "inputs": {
                    "vae": ["5", 0],
                    "samples": ["4", 0]
                }
            },
            "7": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["6", 0],
                    "frame_rate": 25,
                    "format": "video/h264-mp4",
                    "save_output": True
                }
            }
        }
    
    def _extract_images(self, client: RemoteComfyUIClient, history: dict) -> torch.Tensor:
        """Extract images from remote execution result"""
        outputs = history.get('outputs', {})
        
        # Find video output node
        for node_id, node_output in outputs.items():
            if 'gifs' in node_output:
                # Get video frames
                gif_data = node_output['gifs'][0]
                filename = gif_data['filename']
                subfolder = gif_data.get('subfolder', '')
                
                # Download video/gif
                img = client.get_image(filename, subfolder, 'output')
                
                # Convert to tensor
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)
                
                # Add batch dimension if needed
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                
                return img_tensor
        
        raise ValueError("No output images found in remote execution result")


class RemoteT5Encoder(ComfyNodeABC):
    """
    Remote T5 Text Encoder - Execute T5 encoding on remote GPU
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "remote_server": (IO.STRING, {
                    "default": "http://127.0.0.1:8188",
                    "tooltip": "Remote ComfyUI server URL"
                }),
                "positive_prompt": (IO.STRING, {
                    "default": "",
                    "multiline": True
                }),
                "negative_prompt": (IO.STRING, {
                    "default": "",
                    "multiline": True
                }),
                "model_name": (IO.STRING, {
                    "default": "umt5-xxl-enc",
                    "tooltip": "T5 model name on remote server"
                }),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "encode_remote"
    CATEGORY = "WanVideoWrapper/Remote"
    DESCRIPTION = "Encode text using remote T5 model"
    
    async def encode_remote(
        self,
        remote_server: str,
        positive_prompt: str,
        negative_prompt: str,
        model_name: str,
        precision: str,
        **kwargs
    ):
        server = self._resolve_server(remote_server)
        client = RemoteComfyUIClient(server)
        
        # Build minimal workflow for T5 encoding
        workflow = {
            "1": {
                "class_type": "WanVideoTextEncodeCached",
                "inputs": {
                    "model_name": model_name,
                    "precision": precision,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt
                }
            }
        }
        
        print(f"[RemoteT5] Encoding on {server}")
        prompt_id = client.queue_prompt(workflow)
        history = client.wait_for_completion(prompt_id, timeout=300)
        
        # Extract embeddings (would need custom output format)
        # For now, return placeholder
        print("[RemoteT5] Encoding complete")
        
        # TODO: Implement proper embedding extraction
        return ({
            "prompt_embeds": None,
            "negative_prompt_embeds": None
        },)

    def _resolve_server(self, remote_server: str) -> str:
        s = (remote_server or "").strip()
        if s:
            return s
        eng = get_remote_engine()
        if eng.get("enabled") and eng.get("base"):
            return str(eng["base"]).strip()
        raise ValueError("Remote server not specified and global remote engine is not enabled.")


class RemoteWanVideoGlobal(ComfyNodeABC):
    """
    Remote WanVideo Sampler (Global) - Use global remote engine without specifying URL per node
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model_name": (IO.STRING, {"default": "wan_video_14b.safetensors"}),
                "positive_prompt": (IO.STRING, {"default": "", "multiline": True}),
                "negative_prompt": (IO.STRING, {"default": "", "multiline": True}),
                "width": (IO.INT, {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": (IO.INT, {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "num_frames": (IO.INT, {"default": 81, "min": 1, "max": 500, "step": 4}),
                "steps": (IO.INT, {"default": 30, "min": 1, "max": 100}),
                "cfg": (IO.FLOAT, {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "timeout": (IO.INT, {"default": 3600, "min": 60, "max": 7200}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "WanVideoWrapper/Remote"

    async def run(
        self,
        model_name: str,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        seed: int,
        timeout: int = 3600,
        **kwargs
    ):
        eng = get_remote_engine()
        if not (eng.get("enabled") and eng.get("base")):
            raise ValueError("Global remote engine is not enabled or base URL is missing.")
        client = RemoteComfyUIClient(str(eng["base"]).strip())

        workflow = RemoteWanVideoSampler._build_workflow(self,
            model_name=model_name,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed,
        )

        print(f"[RemoteWanVideoGlobal] Submitting to {eng['base']}")
        pid = client.queue_prompt(workflow)
        history = client.wait_for_completion(pid, timeout)
        images = RemoteWanVideoSampler._extract_images(self, client, history)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "RemoteWanVideoSampler": RemoteWanVideoSampler,
    "RemoteT5Encoder": RemoteT5Encoder,
    "RemoteWanVideoGlobal": RemoteWanVideoGlobal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteWanVideoSampler": "Remote WanVideo Sampler",
    "RemoteT5Encoder": "Remote T5 Encoder",
    "RemoteWanVideoGlobal": "Remote WanVideo (Global)",
}

import io
import json
import time
from typing import Any, Dict, Optional, List

import requests
import numpy as np
import torch
from PIL import Image, ImageSequence
import folder_paths

from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict

from .remote_global import get_remote_engine
from .wanvideo.schedulers import scheduler_list

try:
    # Reuse client if available
    from .remote_comfyui_node import RemoteComfyUIClient
except Exception:
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
        def get_file(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
            url = f"{self.server_url}/view"
            params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.content
        def wait_for_completion(self, prompt_id: str, timeout: int = 3600) -> dict:
            start_time = time.time()
            while time.time() - start_time < timeout:
                history = self.get_history(prompt_id)
                if history and history.get('status', {}).get('completed', False):
                    return history
                time.sleep(2)
            raise TimeoutError(f"Task {prompt_id} timed out after {timeout}s")


class WanVideoRemoteAllInOne(ComfyNodeABC):
    """
    All-in-one WanVideo remote panel.
    Build a full WanVideo workflow on the remote server and execute it.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        # Build dropdown options with local scan and optional remote intersection
        def list_local(folder: str) -> List[str]:
            try:
                return folder_paths.get_filename_list(folder)
            except Exception:
                return []

        def list_remote(folder: str, base: Optional[str]) -> List[str]:
            if not base:
                return []
            try:
                url = base.rstrip('/') + f"/models/{folder}"
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    return r.json() or []
            except Exception:
                pass
            return []

        def intersect_lists(local_list: List[str], remote_list: List[str]) -> List[str]:
            if not remote_list:
                return local_list
            rs = set(remote_list)
            return [x for x in local_list if x in rs]

        eng = get_remote_engine()
        remote_base = None
        if eng.get("enabled") and eng.get("base"):
            remote_base = str(eng["base"]).strip()

        # Model files (union of diffusion_models and unet_gguf)
        local_models = list_local("unet_gguf") + list_local("diffusion_models")
        remote_models = list_remote("unet_gguf", remote_base) + list_remote("diffusion_models", remote_base)
        model_options = intersect_lists(local_models, remote_models)
        if not model_options:
            model_options = local_models

        # T5 encoders
        local_t5 = list_local("text_encoders")
        remote_t5 = list_remote("text_encoders", remote_base)
        t5_options = intersect_lists(local_t5, remote_t5) or local_t5 or ["umt5-xxl-enc"]

        # VAE files (try vae and diffusion_models)
        local_vae = list_local("vae") + list_local("diffusion_models")
        remote_vae = list_remote("vae", remote_base) + list_remote("diffusion_models", remote_base)
        vae_options = intersect_lists(local_vae, remote_vae) or local_vae or ["wan_video_vae.safetensors"]

        # LoRAs
        local_loras = list_local("loras")
        remote_loras = list_remote("loras", remote_base)
        lora_options = intersect_lists(local_loras, remote_loras) or local_loras
        lora_options_full = ["none"] + lora_options if "none" not in lora_options else lora_options

        return {
            "required": {
                # Remote
                "remote_server": ("STRING", {"default": "", "multiline": False, "tooltip": "Remote ComfyUI URL. If empty, use global setting."}),
                # Model loader
                "model_name": (model_options, {"default": (model_options[0] if model_options else "wan_video_14b.safetensors"), "tooltip": "Model filename present on both local and remote when possible"}),
                "base_precision": (["bf16", "fp16", "fp32", "fp16_fast"], {"default": "bf16"}),
                "quantization": (["disabled", "fp8_e4m3fn", "fp8_e4m3fn_scaled", "fp8_e5m2", "fp8_e5m2_scaled", "fp4_experimental", "fp4_scaled", "fp8_e4m3fn_fast", "fp8_e5m2_scaled_fast", "fp4_scaled_fast"], {"default": "disabled"}),
                "attention_mode": (["sdpa", "flash_attn_2", "flash_attn_3", "sageattn", "sageattn_3", "sageattn_3_fp4", "sageattn_3_fp8", "radial_sage_attention"], {"default": "sdpa"}),
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
                # T5
                "t5_model": (t5_options, {"default": (t5_options[0] if t5_options else "umt5-xxl-enc")}),
                "t5_precision": (["bf16", "fp32"], {"default": "bf16"}),
                # Prompts
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                # Image embeds (video shape)
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 500, "step": 1}),
                # Sampler
                "steps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63-1}),
                "scheduler": (scheduler_list, {"default": "unipc"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "force_offload": ("BOOLEAN", {"default": True}),
                # VAE
                "vae_model": (vae_options, {"default": (vae_options[0] if vae_options else "wan_video_vae.safetensors")}),
                # Output
                "frame_rate": ("INT", {"default": 25, "min": 1, "max": 120}),
                "format": (["video/h264-mp4", "video/h265-mp4", "image/gif"], {"default": "video/h264-mp4"}),
                # LoRA simple multi (5 slots)
                "lora_0": (lora_options_full, {"default": "none", "tooltip": "LoRA filename present on both local and remote; choose 'none' to skip"}),
                "strength_0": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001}),
                "lora_1": (lora_options_full, {"default": "none"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001}),
                "lora_2": (lora_options_full, {"default": "none"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001}),
                "lora_3": (lora_options_full, {"default": "none"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001}),
                "lora_4": (lora_options_full, {"default": "none"}),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001}),
                "merge_loras": ("BOOLEAN", {"default": True}),
                "low_mem_load": ("BOOLEAN", {"default": False}),
                # Control
                "timeout": ("INT", {"default": 3600, "min": 60, "max": 86400}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "WanVideoWrapper/Remote"
    DESCRIPTION = "All-in-one remote WanVideo panel"

    def _resolve_server(self, remote_server: str) -> str:
        s = (remote_server or "").strip()
        if s:
            return s
        eng = get_remote_engine()
        if eng.get("enabled") and eng.get("base"):
            return str(eng["base"]).strip()
        raise ValueError("Remote server not specified and global remote engine is not enabled.")

    def _build_workflow(self, p: Dict[str, Any]) -> Dict[str, Any]:
        nodes: Dict[str, Any] = {}
        nid = 1
        def add(node):
            nonlocal nid
            k = str(nid)
            nodes[k] = node
            nid += 1
            return k

        # Choose image embeds node name (compat with older remote versions)
        image_embeds_node = p.get("__cap_image_embeds_node__", "WanVideoImageEmbeds")

        # Model Loader
        m = add({
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": p["model_name"],
                "base_precision": p["base_precision"],
                "quantization": p["quantization"],
                "load_device": p["load_device"],
                "attention_mode": p["attention_mode"],
            }
        })

        # LoRA chain
        last_lora = None
        for idx in range(5):
            name_key = f"lora_{idx}"
            str_key = f"strength_{idx}"
            lname = (p.get(name_key) or "none").strip()
            if not lname or lname == "none":
                continue
            l = add({
                "class_type": "WanVideoLoraSelect",
                "inputs": {
                    "lora": lname,
                    "strength": p.get(str_key, 1.0),
                    **({"prev_lora": [last_lora, 0]} if last_lora else {}),
                    "low_mem_load": p.get("low_mem_load", False),
                    "merge_loras": p.get("merge_loras", True),
                }
            })
            last_lora = l
        # Attach lora to model loader if any
        if last_lora:
            nodes[m]["inputs"]["lora"] = [last_lora, 0]

        # Text Encode
        te = add({
            "class_type": "WanVideoTextEncodeCached",
            "inputs": {
                "model_name": p["t5_model"],
                "precision": p["t5_precision"],
                "positive_prompt": p["positive_prompt"],
                "negative_prompt": p["negative_prompt"],
                "use_disk_cache": True,
                "device": "gpu",
            }
        })

        # Image embeds
        im = add({
            "class_type": image_embeds_node,
            "inputs": {
                "width": p["width"],
                "height": p["height"],
                "num_frames": p["num_frames"],
            }
        })

        # Sampler
        sp = add({
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": [m, 0],
                "text_embeds": [te, 0],
                "image_embeds": [im, 0],
                "steps": p["steps"],
                "cfg": p["cfg"],
                "seed": p["seed"],
                "shift": p["shift"],
                "scheduler": p["scheduler"],
                "riflex_freq_index": 0,
                "force_offload": p["force_offload"],
            }
        })

        # VAE
        v = add({
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "model_name": p["vae_model"],
                "precision": "bf16",
            }
        })
        vd = add({
            "class_type": p.get("__cap_vae_decode_node__", "WanVideoVAEDecode"),
            "inputs": {
                "vae": [v, 0],
                "samples": [sp, 0],
            }
        })

        # Combine
        vc = add({
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": [vd, 0],
                "frame_rate": p["frame_rate"],
                "format": p["format"],
                "save_output": True,
            }
        })

        return nodes

    def _extract_images(self, client: RemoteComfyUIClient, history: dict) -> torch.Tensor:
        outputs = history.get('outputs', {})
        # prefer images, then gifs
        imgs: List[torch.Tensor] = []
        for _, out in outputs.items():
            if isinstance(out, dict) and "images" in out:
                for it in out["images"]:
                    fn = it.get("filename"); sf = it.get("subfolder", ""); tp = it.get("type", "output")
                    content = client.get_file(fn, sf, tp)
                    pil = Image.open(io.BytesIO(content)).convert("RGB")
                    arr = np.array(pil).astype(np.float32) / 255.0
                    imgs.append(torch.from_numpy(arr))
            if isinstance(out, dict) and "gifs" in out:
                for it in out["gifs"]:
                    fn = it.get("filename"); sf = it.get("subfolder", ""); tp = it.get("type", "output")
                    content = client.get_file(fn, sf, tp)
                    try:
                        with Image.open(io.BytesIO(content)) as im:
                            for frame in ImageSequence.Iterator(im):
                                fr = frame.convert("RGB")
                                arr = np.array(fr).astype(np.float32) / 255.0
                                imgs.append(torch.from_numpy(arr))
                    except Exception:
                        pass
        if not imgs:
            # fallback empty image
            arr = np.zeros((1, 1, 3), dtype=np.float32)
            return torch.from_numpy(arr).unsqueeze(0)
        batch = torch.stack(imgs, dim=0)
        return batch

    def run(self,
            remote_server: str,
            model_name: str,
            base_precision: str,
            quantization: str,
            attention_mode: str,
            load_device: str,
            t5_model: str,
            t5_precision: str,
            positive_prompt: str,
            negative_prompt: str,
            width: int,
            height: int,
            num_frames: int,
            steps: int,
            cfg: float,
            seed: int,
            scheduler: str,
            shift: float,
            force_offload: bool,
            vae_model: str,
            frame_rate: int,
            format: str,
            lora_0: str,
            strength_0: float,
            lora_1: str,
            strength_1: float,
            lora_2: str,
            strength_2: float,
            lora_3: str,
            strength_3: float,
            lora_4: str,
            strength_4: float,
            merge_loras: bool,
            low_mem_load: bool,
            timeout: int,
            ):
        server = self._resolve_server(remote_server)
        client = RemoteComfyUIClient(server)

        # Preflight: probe remote node availability and set fallbacks
        try:
            oi = requests.get(server.rstrip('/') + "/object_info", timeout=5)
            available = set(oi.json().keys()) if oi.status_code == 200 else set()
        except Exception:
            available = set()

        # Required nodes on remote
        base_required = [
            "WanVideoModelLoader",
            "WanVideoTextEncodeCached",
            "WanVideoSampler",
            "WanVideoVAELoader",
            "VHS_VideoCombine",
        ]
        missing = [n for n in base_required if n not in available]

        # Image embeds compatibility (fuzzy)
        image_embeds_node = None
        candidates_img = [
            "WanVideoImageEmbeds",
            "WanVideoInitEmbeds",
        ]
        for c in candidates_img:
            if c in available:
                image_embeds_node = c
                break
        if image_embeds_node is None:
            # fuzzy: any node that contains both 'WanVideo' and 'Embeds'
            for name in available:
                n = name.lower()
                if ("wanvideo" in n) and ("embed" in n) and ("image" in n or "init" in n):
                    image_embeds_node = name
                    break
        if image_embeds_node is None:
            missing.append("WanVideoImageEmbeds (or WanVideoInitEmbeds)")

        # VAE decode compatibility (fuzzy)
        vae_decode_node = None
        candidates_vae = [
            "WanVideoVAEDecode",
            "VAEDecode",
            "VAE_Decode",
            "VAE Decode",
        ]
        for c in candidates_vae:
            if c in available:
                vae_decode_node = c
                break
        if vae_decode_node is None:
            # fuzzy: any node that contains 'VAE' and 'Decode'
            for name in available:
                n = name.lower()
                if ("vae" in n) and ("decode" in n):
                    vae_decode_node = name
                    break
        if vae_decode_node is None:
            missing.append("WanVideoVAEDecode (or VAEDecode)")

        if missing:
            raise RuntimeError(
                "Remote WanVideo environment incomplete. Missing nodes on remote: " + ", ".join(missing) +
                ". Please install/enable ComfyUI-WanVideoWrapper and VHS on the remote and restart ComfyUI."
            )

        params = locals().copy()
        params["__cap_image_embeds_node__"] = image_embeds_node
        params["__cap_vae_decode_node__"] = vae_decode_node
        # Build and submit
        workflow = self._build_workflow(params)
        pid = client.queue_prompt(workflow)
        history = client.wait_for_completion(pid, timeout)
        images = self._extract_images(client, history)
        # Add batch dim if needed (B,H,W,C)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "WanVideoRemoteAllInOne": WanVideoRemoteAllInOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoRemoteAllInOne": "Remote WanVideo Panel (Global)",
}

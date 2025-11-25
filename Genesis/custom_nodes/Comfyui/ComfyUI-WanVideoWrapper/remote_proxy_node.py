import json
import time
import threading
import io
from typing import Any, Dict, Optional, List

import requests
import torch
import numpy as np
from PIL import Image, ImageSequence

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.utils import ProgressBar
import folder_paths

from .remote_global import get_remote_engine

# --- Server hook (global) ---
_hook_installed = False

def _install_on_prompt_hook():
    global _hook_installed
    if _hook_installed:
        return
    try:
        from server import PromptServer
    except Exception:
        return

    # If instance exists, register immediately; otherwise monkey-patch __init__
    def register(ps):
        try:
            ps.add_on_prompt_handler(_on_prompt_handler)
        except Exception:
            pass

    if getattr(PromptServer, "instance", None) is not None:
        try:
            register(PromptServer.instance)
            _hook_installed = True
            return
        except Exception:
            pass

    orig_init = PromptServer.__init__

    def patched_init(self, loop):
        orig_init(self, loop)
        try:
            register(self)
        finally:
            pass

    PromptServer.__init__ = patched_init
    _hook_installed = True


def _on_prompt_handler(json_data: Dict[str, Any]) -> Dict[str, Any]:
    eng = get_remote_engine()
    if not (eng.get("enabled") and eng.get("base")):
        return json_data

    # Wrap original request into a single proxy node prompt
    base = str(eng["base"]).strip()
    original_payload = json_data.copy()

    proxy_prompt: Dict[str, Any] = {
        "0": {
            "class_type": "RemotePromptExecutor",
            "inputs": {
                "remote_server": base,
                "payload_json": json.dumps(original_payload, ensure_ascii=False),
                "timeout": 3600,
            },
        }
    }

    new_json = {
        "number": original_payload.get("number", 0),
        "front": original_payload.get("front", False),
        "prompt": proxy_prompt,
        "prompt_id": original_payload.get("prompt_id", None) or original_payload.get("client_id", None) or None,
        "extra_data": original_payload.get("extra_data", {}),
        "client_id": original_payload.get("client_id", None),
    }

    # Mark as proxied (for debugging)
    if "extra_data" not in new_json:
        new_json["extra_data"] = {}
    new_json["extra_data"]["wanvideo_remote_proxy"] = True

    return new_json


class RemotePromptExecutor(ComfyNodeABC):
    """
    Execute the original ComfyUI prompt on a remote ComfyUI server and fetch results back.
    This node is designed to be injected globally by the on_prompt hook when remote is enabled.
    """

    API_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "remote_server": (IO.STRING, {"default": "http://127.0.0.1:8188"}),
                "payload_json": (IO.STRING, {"default": "", "multiline": True}),
                "timeout": (IO.INT, {"default": 3600, "min": 60, "max": 86400}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "WanVideoWrapper/Remote"
    DESCRIPTION = "Globally proxy prompts to a remote ComfyUI server"

    def run(self, remote_server: str, payload_json: str, timeout: int = 3600):
        server = remote_server.strip()
        if not server:
            eng = get_remote_engine()
            if not (eng.get("enabled") and eng.get("base")):
                raise ValueError("Remote is not enabled and no server specified")
            server = str(eng["base"]).strip()

        try:
            payload = json.loads(payload_json)
        except Exception:
            raise ValueError("Invalid payload_json: not a valid JSON")

        # Ensure we forward the original prompt as-is
        if "prompt" not in payload:
            raise ValueError("payload_json must contain a 'prompt' field")

        # Submit to remote
        pbar = ProgressBar(3)
        submit_url = server.rstrip("/") + "/prompt"
        r = requests.post(submit_url, json=payload, timeout=30)
        r.raise_for_status()
        pbar.update(1)

        remote_pid = r.json().get("prompt_id")
        if not remote_pid:
            raise RuntimeError("Remote did not return prompt_id")

        # Poll history
        history_url = server.rstrip("/") + f"/history/{remote_pid}"
        start = time.time()
        images: List[torch.Tensor] = []
        while True:
            hr = requests.get(history_url, timeout=30)
            hr.raise_for_status()
            data = hr.json().get(remote_pid)
            if data and data.get("status", {}).get("completed"):
                outputs = data.get("outputs", {})
                images = self._fetch_outputs_as_images(server, outputs)
                break
            if time.time() - start > timeout:
                raise TimeoutError("Remote task timed out")
            time.sleep(2)
        pbar.update(1)

        if not images:
            # Return a 1x1 black image placeholder to satisfy output contract
            arr = np.zeros((1, 1, 1, 3), dtype=np.float32)
            return (torch.from_numpy(arr),)

        # Stack to B,H,W,C float32 [0,1]
        batch = torch.stack(images, dim=0)
        pbar.update(1)
        return (batch,)

    def _fetch_outputs_as_images(self, server: str, outputs: Dict[str, Any]) -> List[torch.Tensor]:
        imgs: List[torch.Tensor] = []
        # Prefer 'images' lists from remote outputs
        for node_id, out in outputs.items():
            # images
            if isinstance(out, dict) and "images" in out:
                for it in out["images"]:
                    fn = it.get("filename")
                    sf = it.get("subfolder", "")
                    tp = it.get("type", "output")
                    img = self._download_image(server, fn, sf, tp)
                    if img is not None:
                        imgs.append(img)
            # gifs
            if isinstance(out, dict) and "gifs" in out:
                for it in out["gifs"]:
                    fn = it.get("filename")
                    sf = it.get("subfolder", "")
                    tp = it.get("type", "output")
                    frames = self._download_gif_frames(server, fn, sf, tp)
                    imgs.extend(frames)
        return imgs

    def _download_image(self, server: str, filename: Optional[str], subfolder: str, folder_type: str) -> Optional[torch.Tensor]:
        if not filename:
            return None
        url = server.rstrip("/") + "/view"
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            return None
        pil = Image.open(io.BytesIO(r.content))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def _download_gif_frames(self, server: str, filename: Optional[str], subfolder: str, folder_type: str) -> List[torch.Tensor]:
        frames: List[torch.Tensor] = []
        if not filename:
            return frames
        url = server.rstrip("/") + "/view"
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 200:
            return frames
        try:
            with Image.open(io.BytesIO(r.content)) as im:
                for frame in ImageSequence.Iterator(im):
                    fr = frame.convert("RGB")
                    arr = np.array(fr).astype(np.float32) / 255.0
                    frames.append(torch.from_numpy(arr))
        except Exception:
            pass
        return frames


# Ensure hook is installed when module is imported
_install_on_prompt_hook()


NODE_CLASS_MAPPINGS = {
    "RemotePromptExecutor": RemotePromptExecutor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotePromptExecutor": "Remote Prompt Executor (Global)",
}

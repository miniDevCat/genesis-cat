import os
import json
from typing import Dict
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "remote_engine.json")
_remote_engine: Dict[str, str | bool | None] = {"enabled": False, "base": None}


def _load_from_env() -> None:
    en = os.environ.get("WANREMOTE_ENABLED", None)
    base = os.environ.get("WANREMOTE_BASE", None)
    if en is not None:
        _remote_engine["enabled"] = str(en).strip().lower() in {"1", "true", "yes", "on"}
    if base:
        _remote_engine["base"] = base.strip()


def _load_from_file() -> None:
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _remote_engine["enabled"] = bool(data.get("enabled", _remote_engine["enabled"]))
                b = data.get("base", None)
                if b:
                    _remote_engine["base"] = str(b).strip()
    except Exception:
        pass


def save_remote_engine() -> None:
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"enabled": _remote_engine["enabled"], "base": _remote_engine["base"]}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def set_remote_engine(enabled: bool, base: str | None) -> None:
    _remote_engine["enabled"] = bool(enabled)
    _remote_engine["base"] = (base or "").strip() or None


def get_remote_engine() -> Dict[str, str | bool | None]:
    return _remote_engine


# Initialize from env and file at import time
_load_from_env()
_load_from_file()


class RemoteEngineGlobal(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "enable": (IO.BOOLEAN, {"default": True}),
                "remote_server": (IO.STRING, {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    RETURN_NAMES = ("enabled",)
    FUNCTION = "set_global"
    CATEGORY = "WanVideoWrapper/Remote"

    def set_global(self, enable: bool, remote_server: str):
        set_remote_engine(bool(enable), remote_server)
        save_remote_engine()
        return (bool(get_remote_engine()["enabled"]),)

NODE_CLASS_MAPPINGS = {
    "RemoteEngineGlobal": RemoteEngineGlobal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteEngineGlobal": "Remote Engine (Global)",
}

import folder_paths
import torch
import torch.nn.functional as F
import os
import json
import torchaudio
import math

from comfy.utils import load_torch_file, common_upscale
import comfy.model_management as mm

from accelerate import init_empty_weights
from ..utils import set_module_tensor_to_device, log
from ..nodes import WanVideoEncodeLatentBatch

script_directory = os.path.dirname(os.path.abspath(__file__))
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

def linear_interpolation_fps(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)  # [1, C, T]
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

def get_audio_emb_window(audio_emb, frame_num, frame0_idx, audio_shift=2):
    zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    iter_ = 1 + (frame_num - 1) // 4
    audio_emb_wind = []
    for lt_i in range(iter_):
        if lt_i == 0:
            st = frame0_idx + lt_i - 2
            ed = frame0_idx + lt_i + 3
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
            wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
        else:
            st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
            ed = frame0_idx + 1 + 4 * lt_i + audio_shift
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
        audio_emb_wind.append(wind_feat)
    audio_emb_wind = torch.stack(audio_emb_wind, dim=0)

    return audio_emb_wind, ed - audio_shift

class HuMoEmbedsFlexible:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": -1, "max": 10000, "step": 1, "tooltip": "The exact frame count to generate - no constraints applied"}),
            "width": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 16}),
            "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the audio conditioning"}),
            "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "When not 1.0, an extra model pass without audio conditioning is done: slower inference but more motion is allowed"}),
            "audio_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The percent of the video to start applying audio conditioning"}),
            "audio_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The percent of the video to stop applying audio conditioning"})
        },
            "optional" : {
                "whisper_model": ("WHISPERMODEL",),
                "vae": ("WANVAE", ),
                "reference_images": ("IMAGE", {"tooltip": "reference images for the humo model"}),
                "audio": ("AUDIO",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds", )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, audio_scale, audio_cfg_scale, audio_start_percent, audio_end_percent, whisper_model=None, vae=None, reference_images=None, audio=None, tiled_vae=False):
        if reference_images is not None and vae is None:
            raise ValueError("VAE is required when reference images are provided")
        if whisper_model is None and audio is not None:
            raise ValueError("Whisper model is required when audio is provided")
        
        if whisper_model is not None:
            model = whisper_model["model"]
            feature_extractor = whisper_model["feature_extractor"]
            dtype = whisper_model["dtype"]
        else:
            model = None
            feature_extractor = None
            dtype = torch.float16

        sampling_rate = 16000

        if audio is not None and model is not None:
            audio_input = audio["waveform"][0]
            sample_rate = audio["sample_rate"]

            if sample_rate != sampling_rate:
                audio_input = torchaudio.functional.resample(audio_input, sample_rate, sampling_rate)
            if audio_input.shape[1] == 2:
                audio_input = audio_input.mean(dim=0, keepdim=False)
            else:
                audio_input = audio_input[0]

            model.to(device)
            audio_len = len(audio_input) // 640

            # feature extraction
            audio_features = []
            window = 750*640
            for i in range(0, len(audio_input), window):
                audio_feature = feature_extractor(audio_input[i:i+window], sampling_rate=sampling_rate, return_tensors="pt").input_features
                audio_features.append(audio_feature)
            audio_features = torch.cat(audio_features, dim=-1).to(device, dtype)

            # preprocess
            window = 3000
            audio_prompts = []
            for i in range(0, audio_features.shape[-1], window):
                audio_prompt = model.encoder(audio_features[:,:,i:i+window], output_hidden_states=True).hidden_states
                audio_prompt = torch.stack(audio_prompt, dim=2)
                audio_prompts.append(audio_prompt)

            model.to(offload_device)

            audio_prompts = torch.cat(audio_prompts, dim=1)
            audio_prompts = audio_prompts[:,:audio_len*2]

            feat0 = linear_interpolation_fps(audio_prompts[:, :, 0: 8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation_fps(audio_prompts[:, :, 8: 16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation_fps(audio_prompts[:, :, 16: 24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation_fps(audio_prompts[:, :, 24: 32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation_fps(audio_prompts[:, :, 32], 50, 25)
            audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T, 5, 1280]
        else:
            audio_emb = torch.zeros(num_frames, 5, 1280, device=device)
            audio_len = num_frames
            
        # Use exact frame count - no 4n+1 constraint
        pixel_frame_num = num_frames if num_frames != -1 else audio_len
        
        # Calculate latent frames using ceiling division to ensure sufficient capacity
        latent_frame_num = math.ceil(pixel_frame_num / 4)
        
        log.info(f"HuMo Flexible set to generate exactly {pixel_frame_num} frames (latent: {latent_frame_num})")

        num_refs = 0
        if reference_images is not None:
            if reference_images.shape[1] != height or reference_images.shape[2] != width:
                reference_images_in = common_upscale(reference_images.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            else:
                reference_images_in = reference_images
            samples, = WanVideoEncodeLatentBatch.encode(self, vae, reference_images_in, tiled_vae, None, None, None, None)
            samples = samples["samples"].transpose(0, 2).squeeze(0)
            num_refs = samples.shape[1]

        if vae is not None:
            vae.to(device)
            zero_frames = torch.zeros(1, 3, pixel_frame_num + 4*num_refs, height, width, device=device, dtype=vae.dtype)
            zero_latents = vae.encode(zero_frames, device=device, tiled=tiled_vae)[0].to(offload_device)
            
            vae.to(offload_device)
            mm.soft_empty_cache()
        else:
            # Create dummy zero latents if no VAE provided
            zero_latents = torch.zeros(16, latent_frame_num + num_refs, height // 8, width // 8, device=offload_device, dtype=torch.float16)

        target_shape = (16, latent_frame_num + num_refs, height // 8, width // 8)

        mask = torch.ones(4, target_shape[1], target_shape[2], target_shape[3], device=offload_device, dtype=zero_latents.dtype)
        if reference_images is not None:
            mask[:,:-num_refs] = 0
            image_cond = torch.cat([zero_latents[:, :(target_shape[1]-num_refs)], samples], dim=1)
        else:
            image_cond = zero_latents
            mask = torch.zeros_like(mask)
        image_cond = torch.cat([mask, image_cond], dim=0)
        image_cond_neg = torch.cat([mask, zero_latents], dim=0)

        embeds = {
            "humo_audio_emb": audio_emb,
            "humo_audio_emb_neg": torch.zeros_like(audio_emb, dtype=audio_emb.dtype, device=audio_emb.device),
            "humo_image_cond": image_cond,
            "humo_image_cond_neg": image_cond_neg,
            "humo_reference_count": num_refs,
            "target_shape": target_shape,
            "num_frames": pixel_frame_num,
            "requested_num_frames": pixel_frame_num,
            "force_exact_frames": True,  # Always force exact frames for this node
            "humo_audio_scale": audio_scale,
            "humo_audio_cfg_scale": audio_cfg_scale,
            "humo_start_percent": audio_start_percent,
            "humo_end_percent": audio_end_percent,
        }
        
        return (embeds, )

NODE_CLASS_MAPPINGS = {
    "HuMoEmbedsFlexible": HuMoEmbedsFlexible,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuMoEmbedsFlexible": "HuMo Embeds Flexible",
}

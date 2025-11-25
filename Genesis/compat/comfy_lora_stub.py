"""
ComfyUI comfy.lora module stub
Provides LoRA calculation functions
"""

import torch
import logging

logger = logging.getLogger(__name__)


def calculate_weight(patches, temp_weight, key):
    """
    Apply LoRA patches to weight
    patches: list of (strength_patch, patch, strength_model, offset, function)
    """
    logger.debug(f"[calculate_weight] Called for key: {key}")
    logger.debug(f"[calculate_weight] Number of patches: {len(patches)}")
    logger.debug(f"[calculate_weight] temp_weight shape: {temp_weight.shape}")
    
    for idx, (strength_patch, patch, strength_model, offset, function) in enumerate(patches):
        logger.debug(f"[calculate_weight] Patch {idx}: strength_patch={strength_patch}, strength_model={strength_model}")
        logger.debug(f"[calculate_weight] Patch type: {type(patch)}")
        
        if function is not None:
            # Custom function
            logger.debug(f"[calculate_weight] Using custom function")
            temp_weight = function(temp_weight, patch, strength_patch, strength_model)
        else:
            # Standard LoRA application
            alpha = strength_patch * strength_model
            logger.debug(f"[calculate_weight] Alpha (combined strength): {alpha}")
            
            if isinstance(patch, torch.Tensor):
                # Direct tensor patch
                logger.debug(f"[calculate_weight] Direct tensor patch, shape: {patch.shape}")
                if offset is not None:
                    temp_weight[offset] += alpha * patch.to(temp_weight.device, temp_weight.dtype)
                else:
                    temp_weight += alpha * patch.to(temp_weight.device, temp_weight.dtype)
            elif isinstance(patch, tuple):
                logger.debug(f"[calculate_weight] Tuple patch, length: {len(patch)}")
                if len(patch) >= 2:
                    # LoRA format: (lora_up, lora_down) or (lora_up, lora_down, alpha)
                    lora_up = patch[0]
                    lora_down = patch[1]
                    logger.debug(f"[calculate_weight] LoRA up shape: {lora_up.shape}, down shape: {lora_down.shape}")
                    
                    # Check if there's a pre-computed alpha in the patch
                    if len(patch) > 2 and isinstance(patch[2], (int, float)):
                        patch_alpha = patch[2]
                        logger.debug(f"[calculate_weight] Using patch alpha: {patch_alpha}")
                        alpha = alpha * patch_alpha
                    
                    # weight = weight + alpha * (lora_up @ lora_down)
                    lora_diff = torch.mm(
                        lora_up.to(temp_weight.device, temp_weight.dtype),
                        lora_down.to(temp_weight.device, temp_weight.dtype)
                    )
                    logger.debug(f"[calculate_weight] LoRA diff shape: {lora_diff.shape}")
                    
                    if offset is not None:
                        temp_weight[offset] += alpha * lora_diff
                    else:
                        temp_weight += alpha * lora_diff
                else:
                    logger.warning(f"[calculate_weight] Unknown tuple patch format with length {len(patch)}")
            else:
                logger.warning(f"[calculate_weight] Unknown patch type: {type(patch)}")
    
    logger.debug(f"[calculate_weight] Final weight shape: {temp_weight.shape}")
    return temp_weight


def load_lora(lora_sd, key_map=None):
    """
    Load LoRA from state dict
    This is called by WanVideoWrapper's load_lora_for_models_mod
    """
    logger.info(f"[load_lora] Loading LoRA with {len(lora_sd)} keys")
    
    # Group LoRA keys by base name
    lora_patches = {}
    
    for key in lora_sd:
        # LoRA keys typically end with .lora_up.weight or .lora_down.weight
        if '.lora_up.weight' in key:
            base_key = key.replace('.lora_up.weight', '.weight')
            if base_key not in lora_patches:
                lora_patches[base_key] = {}
            lora_patches[base_key]['up'] = lora_sd[key]
        elif '.lora_down.weight' in key:
            base_key = key.replace('.lora_down.weight', '.weight')
            if base_key not in lora_patches:
                lora_patches[base_key] = {}
            lora_patches[base_key]['down'] = lora_sd[key]
        elif '.lora_delta' in key:
            # Direct delta patch
            base_key = key.replace('.lora_delta', '')
            if base_key not in lora_patches:
                lora_patches[base_key] = {}
            lora_patches[base_key]['delta'] = lora_sd[key]
        elif '.alpha' in key:
            # Alpha value
            base_key = key.replace('.alpha', '.weight')
            if base_key not in lora_patches:
                lora_patches[base_key] = {}
            lora_patches[base_key]['alpha'] = lora_sd[key]
    
    # Convert to patch format: (lora_up, lora_down) or direct tensor
    result = {}
    for base_key, patch_data in lora_patches.items():
        if 'up' in patch_data and 'down' in patch_data:
            # Standard LoRA format
            if 'alpha' in patch_data:
                result[base_key] = (patch_data['up'], patch_data['down'], patch_data['alpha'])
            else:
                result[base_key] = (patch_data['up'], patch_data['down'])
        elif 'delta' in patch_data:
            # Direct delta
            result[base_key] = patch_data['delta']
    
    logger.info(f"[load_lora] Converted to {len(result)} patches")
    return result

import torch
from .fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .basic_flowmatch import FlowMatchScheduler
from .flowmatch_pusa import FlowMatchSchedulerPusa
from .flowmatch_res_multistep import FlowMatchSchedulerResMultistep
from .scheduling_flow_match_lcm import FlowMatchLCMScheduler
from .flowmatch_sa_ode_stable import FlowMatchSAODEStableScheduler
from .humo_lcm_integration import get_humo_lcm_scheduler
from .fm_rcm import rCMFlowMatchScheduler
try:
    from .flowmatch_frame_euler_d import FlowMatchFrameEulerDScheduler
except ImportError:
    FlowMatchFrameEulerDScheduler = None
try:
    from .iching_wuxing_scheduler_core import IChingWuxingScheduler
except ImportError:
    IChingWuxingScheduler = None
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
try:
    from ...utils import log
except ImportError:
    class SimpleLog:
        def info(self, msg): print(f"INFO: {msg}")
    log = SimpleLog()

scheduler_list = [
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
    "flowmatch_frame_euler_d",
    "flowmatch_sa_ode_stable",
    "sa_ode_stable/lowstep",
    "humo_lcm",
    "multitalk",
    "iching/wuxing",
    "iching/wuxing-strong",
    "iching/wuxing-stable",
    "iching/wuxing-smooth",
    "iching/wuxing-clean",
    "iching/wuxing-sharp",
    "iching/wuxing-lowstep",
    "rcm"
]

def get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer_dim=5120, flowedit_args=None, denoise_strength=1.0, sigmas=None, log_timesteps=False):
    timesteps = None
    if 'unipc' in scheduler:
        sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.sigmas = sigmas.to(device)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)

    elif scheduler in ['euler/beta', 'euler']:
        sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
        if flowedit_args:
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=get_sampling_sigmas(steps, shift))
        else:
            sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif 'dpm' in scheduler:
        if 'sde' in scheduler:
            algorithm_type = "sde-dpmsolver++"
        else:
            algorithm_type = "dpmsolver++"
        sample_scheduler = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type=algorithm_type)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.sigmas = sigmas.to(device)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)
    elif scheduler == 'deis':
        sample_scheduler = DEISMultistepScheduler(use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=shift)
        sample_scheduler.set_timesteps(steps, device=device)
        sample_scheduler.sigmas[-1] = 1e-6
    elif 'lcm' in scheduler:
        sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
        sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif 'flowmatch_causvid' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        if transformer_dim == 5120:
            denoising_list = [999, 934, 862, 756, 603, 410, 250, 140, 74]
        else:
            if steps != 4:
                raise ValueError("CausVid 1.3B schedule is only for 4 steps")
            denoising_list = [1000, 750, 500, 250]
        sample_scheduler = FlowMatchScheduler(num_inference_steps=steps, shift=shift, sigma_min=0, extra_one_step=True)
        sample_scheduler.timesteps = torch.tensor(denoising_list)[:steps].to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_distill' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        sample_scheduler = FlowMatchScheduler(
            shift=shift, sigma_min=0.0, extra_one_step=True
        )
        sample_scheduler.set_timesteps(1000, training=True)
    
        denoising_step_list = torch.tensor([999, 750, 500, 250] , dtype=torch.long)
        temp_timesteps = torch.cat((sample_scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        denoising_step_list = temp_timesteps[1000 - denoising_step_list]

        
        if steps != 4:
            raise ValueError("This scheduler is only for 4 steps")
        
        sample_scheduler.timesteps = denoising_step_list[:steps].clone().detach().to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_pusa' in scheduler:
        sample_scheduler = FlowMatchSchedulerPusa(shift=shift, sigma_min=0.0, extra_one_step=True)
        sample_scheduler.set_timesteps(steps+1, denoising_strength=denoise_strength, shift=shift,
                                       sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler == 'res_multistep':
        sample_scheduler = FlowMatchSchedulerResMultistep(shift=shift)
        sample_scheduler.set_timesteps(steps, denoising_strength=denoise_strength, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler == 'flowmatch_frame_euler_d':
        if FlowMatchFrameEulerDScheduler is None:
            raise ImportError("FlowMatchFrameEulerDScheduler is not available. The compiled module may be missing.")
        sample_scheduler = FlowMatchFrameEulerDScheduler(shift=shift)
        sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler in ['flowmatch_sa_ode_stable', 'sa_ode_stable/lowstep']:
        sample_scheduler = FlowMatchSAODEStableScheduler(shift=shift)
        sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler == 'humo_lcm':
        sample_scheduler = get_humo_lcm_scheduler(
            steps=steps,
            shift=shift,
            device=device,
            sigmas=sigmas
        )
    elif scheduler.startswith('iching/'):
        if IChingWuxingScheduler is None:
            raise ImportError("IChingWuxingScheduler is not available. The compiled module may be missing.")
        sample_scheduler = IChingWuxingScheduler(mode=scheduler)
        sample_scheduler.set_timesteps(steps, device=device)
    elif scheduler == 'rcm':
        sample_scheduler = rCMFlowMatchScheduler(num_inference_steps=steps)
        sample_scheduler.timesteps = sample_scheduler.timesteps.to(device)
        sample_scheduler.sigmas = sample_scheduler.sigmas.to(device)
    if timesteps is None:
        timesteps = sample_scheduler.timesteps

    steps = len(timesteps)
    if (isinstance(start_step, int) and end_step != -1 and start_step >= end_step) or (not isinstance(start_step, int) and start_step != -1 and end_step >= start_step):
        raise ValueError("start_step must be less than end_step")


    start_idx = 0
    end_idx = len(timesteps) - 1

    if log_timesteps:
        log.info(f"------- Scheduler info -------")
        log.info(f"Total timesteps: {timesteps}")

    if isinstance(start_step, float):
        idxs = (sample_scheduler.sigmas <= start_step).nonzero(as_tuple=True)[0]
        if len(idxs) > 0:
            start_idx = idxs[0].item()
    elif isinstance(start_step, int):
        if start_step > 0:
            start_idx = start_step

    if isinstance(end_step, float):
        idxs = (sample_scheduler.sigmas >= end_step).nonzero(as_tuple=True)[0]
        if len(idxs) > 0:
            end_idx = idxs[-1].item()
    elif isinstance(end_step, int):
        if end_step != -1:
            end_idx = end_step - 1


    timesteps = timesteps[start_idx:end_idx+1]
    sample_scheduler.full_sigmas = sample_scheduler.sigmas.clone()
    sample_scheduler.sigmas = sample_scheduler.sigmas[start_idx:start_idx+len(timesteps)+1]
    
    if log_timesteps:
        log.info(f"Using timesteps: {timesteps}")
        log.info(f"Using sigmas: {sample_scheduler.sigmas}")
        log.info(f"------------------------------")

    if hasattr(sample_scheduler, 'timesteps'):
        sample_scheduler.timesteps = timesteps

    return sample_scheduler, timesteps, start_idx, end_idx

# 导入所有节点类和映射
from .nodes import NODE_CLASS_MAPPINGS as NODES_MAIN, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_MAIN
from .nodes_sampler import NODE_CLASS_MAPPINGS as NODES_SAMPLER, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_SAMPLER
from .nodes_model_loading import NODE_CLASS_MAPPINGS as NODES_MODEL, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_MODEL
from .nodes_utility import NODE_CLASS_MAPPINGS as NODES_UTILITY, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_UTILITY

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(NODES_MAIN)
NODE_CLASS_MAPPINGS.update(NODES_SAMPLER)
NODE_CLASS_MAPPINGS.update(NODES_MODEL)
NODE_CLASS_MAPPINGS.update(NODES_UTILITY)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_MAIN)
NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_SAMPLER)
NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_MODEL)
NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_UTILITY)

# 导入其他子模块的节点
try:
    from .vae_loader import NODE_CLASS_MAPPINGS as NODES_VAE, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_VAE
    NODE_CLASS_MAPPINGS.update(NODES_VAE)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_VAE)
except ImportError:
    pass

try:
    from .cache_methods.nodes_cache import NODE_CLASS_MAPPINGS as NODES_CACHE, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_CACHE
    NODE_CLASS_MAPPINGS.update(NODES_CACHE)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_CACHE)
except ImportError:
    pass

try:
    from .HuMo.nodes import NODE_CLASS_MAPPINGS as NODES_HUMO, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_HUMO
    NODE_CLASS_MAPPINGS.update(NODES_HUMO)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_HUMO)
except ImportError:
    pass

try:
    from .HuMo.nodes_flexible import NODE_CLASS_MAPPINGS as NODES_HUMO_FLEX, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_HUMO_FLEX
    NODE_CLASS_MAPPINGS.update(NODES_HUMO_FLEX)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_HUMO_FLEX)
except ImportError:
    pass

try:
    from .unianimate.nodes import NODE_CLASS_MAPPINGS as NODES_UNIANIM, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_UNIANIM
    NODE_CLASS_MAPPINGS.update(NODES_UNIANIM)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_UNIANIM)
except ImportError:
    pass

try:
    from .fantasytalking.nodes import NODE_CLASS_MAPPINGS as NODES_FANTASY, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_FANTASY
    NODE_CLASS_MAPPINGS.update(NODES_FANTASY)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_FANTASY)
except ImportError:
    pass

try:
    from .uni3c.nodes import NODE_CLASS_MAPPINGS as NODES_UNI3C, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_UNI3C
    NODE_CLASS_MAPPINGS.update(NODES_UNI3C)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_UNI3C)
except ImportError:
    pass

try:
    from .multitalk.nodes import NODE_CLASS_MAPPINGS as NODES_MULTITALK, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_MULTITALK
    NODE_CLASS_MAPPINGS.update(NODES_MULTITALK)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_MULTITALK)
except ImportError:
    pass

try:
    from .ATI.nodes import NODE_CLASS_MAPPINGS as NODES_ATI, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_ATI
    NODE_CLASS_MAPPINGS.update(NODES_ATI)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_ATI)
except ImportError:
    pass

try:
    from .controlnet.nodes import NODE_CLASS_MAPPINGS as NODES_CONTROL, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_CONTROL
    NODE_CLASS_MAPPINGS.update(NODES_CONTROL)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_CONTROL)
except ImportError:
    pass

try:
    from .fun_camera.nodes import NODE_CLASS_MAPPINGS as NODES_FUNCAM, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_FUNCAM
    NODE_CLASS_MAPPINGS.update(NODES_FUNCAM)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_FUNCAM)
except ImportError:
    pass

try:
    from .recammaster.nodes import NODE_CLASS_MAPPINGS as NODES_RECAM, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_RECAM
    NODE_CLASS_MAPPINGS.update(NODES_RECAM)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_RECAM)
except ImportError:
    pass

try:
    from .fantasyportrait.nodes import NODE_CLASS_MAPPINGS as NODES_PORTRAIT, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_PORTRAIT
    NODE_CLASS_MAPPINGS.update(NODES_PORTRAIT)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_PORTRAIT)
except ImportError:
    pass

try:
    from .lynx.nodes import NODE_CLASS_MAPPINGS as NODES_LYNX, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_LYNX
    NODE_CLASS_MAPPINGS.update(NODES_LYNX)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_LYNX)
except ImportError:
    pass

try:
    from .s2v.nodes import NODE_CLASS_MAPPINGS as NODES_S2V, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_S2V
    NODE_CLASS_MAPPINGS.update(NODES_S2V)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_S2V)
except ImportError:
    pass

try:
    from .MTV.nodes import NODE_CLASS_MAPPINGS as NODES_MTV, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_MTV
    NODE_CLASS_MAPPINGS.update(NODES_MTV)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_MTV)
except ImportError:
    pass

try:
    from .FlashVSR.flashvsr_nodes import NODE_CLASS_MAPPINGS as NODES_FLASH, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_FLASH
    NODE_CLASS_MAPPINGS.update(NODES_FLASH)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_FLASH)
except ImportError:
    pass

try:
    from .Ovi.nodes_ovi import NODE_CLASS_MAPPINGS as NODES_OVI, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_OVI
    NODE_CLASS_MAPPINGS.update(NODES_OVI)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_OVI)
except ImportError:
    pass

try:
    from .qwen.qwen import NODE_CLASS_MAPPINGS as NODES_QWEN, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_QWEN
    NODE_CLASS_MAPPINGS.update(NODES_QWEN)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_QWEN)
except ImportError:
    pass

try:
    from .skyreels.nodes import NODE_CLASS_MAPPINGS as NODES_SKY, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_SKY
    NODE_CLASS_MAPPINGS.update(NODES_SKY)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_SKY)
except ImportError:
    pass

try:
    from .remote_allinone_panel import NODE_CLASS_MAPPINGS as NODES_REMOTE_ALL
    NODE_CLASS_MAPPINGS.update(NODES_REMOTE_ALL)
except ImportError:
    pass

try:
    from .remote_comfyui_node import NODE_CLASS_MAPPINGS as NODES_REMOTE_COMFY
    NODE_CLASS_MAPPINGS.update(NODES_REMOTE_COMFY)
except ImportError:
    pass

try:
    from .remote_proxy_node import NODE_CLASS_MAPPINGS as NODES_REMOTE_PROXY
    NODE_CLASS_MAPPINGS.update(NODES_REMOTE_PROXY)
except ImportError:
    pass

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
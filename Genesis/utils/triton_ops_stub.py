"""
Triton ops stub for Windows compatibility
Author: eddy
"""

import sys
from types import ModuleType

# Create triton module if it doesn't exist
if 'triton' not in sys.modules:
    triton = ModuleType('triton')
    sys.modules['triton'] = triton
else:
    triton = sys.modules['triton']

# Create triton.ops module
if not hasattr(triton, 'ops') or 'triton.ops' not in sys.modules:
    ops_module = ModuleType('triton.ops')
    ops_module.__package__ = 'triton'

    # Create matmul_perf_model submodule
    matmul_perf_model = ModuleType('triton.ops.matmul_perf_model')
    matmul_perf_model.__package__ = 'triton.ops'

    def early_config_prune(configs, named_args, **kwargs):
        """Stub for early_config_prune"""
        # Return configs as-is
        return configs if configs else []

    def estimate_matmul_time(**kwargs):
        """Stub for estimate_matmul_time"""
        # Return a dummy time estimate
        return 1.0

    # Add functions to module
    matmul_perf_model.early_config_prune = early_config_prune
    matmul_perf_model.estimate_matmul_time = estimate_matmul_time

    # Link modules together
    ops_module.matmul_perf_model = matmul_perf_model
    triton.ops = ops_module

    # Register in sys.modules
    sys.modules['triton.ops'] = ops_module
    sys.modules['triton.ops.matmul_perf_model'] = matmul_perf_model

    print("[OK] Triton ops stub loaded for Windows compatibility")
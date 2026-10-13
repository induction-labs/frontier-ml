"""The production scoring runtime, also required when freezing CUDA calibration."""

CUDA_RUNTIME = {"torch": "2.5.1", "torch_build": "2.5.1+cu118", "cuda": "11.8",
                "numpy": "1.26.4", "pillow": "11.3.0", "device": "cuda",
                "precision": "float32", "threads": 8}


def check_cuda_runtime():
    import torch
    from vfi_evaluate import runtime_identity
    if runtime_identity("cuda") != CUDA_RUNTIME:
        raise ValueError("Metric worker does not match the frozen CUDA runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Metrics require exactly one H100")
    gpu = torch.cuda.get_device_properties(0)
    if "H100" not in gpu.name.upper() or gpu.total_memory < 75000 * 1024 * 1024:
        raise ValueError("Metrics require an H100 with at least 75000 MiB")
    return {"name": gpu.name, "memory_mib": gpu.total_memory // (1024 * 1024)}

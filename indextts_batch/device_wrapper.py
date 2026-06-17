import torch


def _is_rocm() -> bool:
    """判断当前 PyTorch 是否为 ROCm (AMD GPU) 后端"""
    version_mod = getattr(torch, "version", None)
    if version_mod is None:
        return False
    return getattr(version_mod, "hip", None) is not None


def empty_cache(device: str | torch.device) -> None:
    """跨平台清空 GPU 缓存"""
    device_str = str(device)
    if "cuda" in device_str:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif "mps" in device_str:
        torch.mps.empty_cache()


def get_memory_allocated(device: str | torch.device) -> float:
    """获取当前 PyTorch 分配的显存"""
    device_str = str(device)
    if "cuda" in device_str:
        return torch.cuda.memory_allocated(device)
    elif "mps" in device_str:
        return torch.mps.current_allocated_memory()
    return 0.0


def get_memory_info(
    device: str | torch.device,
) -> tuple[float, float]:
    """获取可用显存和总显存（返回 free_bytes, total_bytes）

    MPS 统一内存架构没有显式的总量限制，total=0 表示无法统计。
    """
    device_str = str(device)
    if "cuda" in device_str:
        return torch.cuda.mem_get_info(device)
    # MPS 无法获取独立显存上限，用当前分配量作为下限
    return 0.0, 0.0


def get_onnx_providers(device: str | torch.device) -> list[str]:
    """根据设备返回对应的 ONNX Execution Provider 列表"""
    device_str = str(device)
    if device_str.startswith("cuda"):
        if _is_rocm():
            return ["ROCMExecutionProvider", "CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device_str.startswith("mps"):
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

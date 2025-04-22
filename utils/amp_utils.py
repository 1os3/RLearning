import torch
from contextlib import contextmanager

# 适配XPU/CPU/CUDA的混合精度上下文管理器
@contextmanager
def autocast_xpu(dtype=torch.bfloat16):
    """
    混合精度上下文，优先支持XPU（intel），否则回退CUDA/CPU。
    dtype: 推荐torch.bfloat16（intel xpu），cuda可用float16。
    """
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        # intel xpu混合精度
        try:
            import intel_extension_for_pytorch as ipex
            with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                yield
        except ImportError:
            # 兼容未安装ipex的情况
            yield
    elif torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            yield
    else:
        # cpu无混合精度
        yield

import cv2
import numpy as np
import torch

def preprocess_image(img, out_size=(640, 360)):
    """
    图像预处理：resize、归一化、转为Tensor，支持高分辨率（默认640x360）
    """
    try:
        # 强制转换为RGB三通道
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, out_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # [3, H, W]
        tensor = torch.from_numpy(img)
        return tensor
    except Exception as e:
        print(f"[VisionUtils] preprocess_image异常: {e}")
        raise

def stack_state(state_list):
    """
    多帧堆叠，适合DQN等
    """
    try:
        return torch.cat(state_list, dim=0)  # [stack, H, W]
    except Exception as e:
        print(f"[VisionUtils] stack_state异常: {e}")
        raise

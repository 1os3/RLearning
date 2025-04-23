import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from lnn.lnn_network import LiquidNeuralNetwork
from config import CONFIG

class VisionLNNAgent(nn.Module):
    """
    视觉强化学习智能体，融合CNN特征提取、LNN动态建模和多分支决策头
    新增：delta_pos（相对初始点位移）、target_pos（目标点坐标）、elapsed_time（已用时间）独立分支输入
    """
    def __init__(self, input_shape=None, action_dim=3, lowdim_dim=9, device=None, lr=None):
        super().__init__()
        try:
            self.device = device or (torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu"))
            # 动态推断输入shape
            if input_shape is None:
                # 默认输入shape由vision_utils.py的preprocess_image决定
                dummy_img = np.zeros((3, 360, 640), dtype=np.float32)  # 假设最大分辨率
                input_shape = dummy_img.shape
            self.input_shape = input_shape
            self.action_dim = action_dim
            self.lowdim_dim = lowdim_dim
            self.action_low = -10  # 可根据环境实际调整
            self.action_high = 10
            # --- 1. 轻量级CNN特征提取 ---
            c, h, w = self.input_shape
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 3)),  # 固定空间输出
                nn.Flatten()
            ).to(self.device)
            # 动态推断CNN输出特征维度
            with torch.no_grad():
                dummy = torch.zeros(1, *self.input_shape, device=self.device)
                feat_dim = self.cnn(dummy).view(1, -1).size(1)
            # --- 1.1 低维状态编码（原有lowdim特征）---
            self.lowdim_encoder = nn.Sequential(
                nn.Linear(lowdim_dim, 16), nn.ReLU(), nn.Linear(16, 8)
            ).to(self.device)
            # --- 1.2 新增：delta_pos分支 ---
            self.delta_pos_encoder = nn.Sequential(
                nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 4)
            ).to(self.device)
            # --- 1.3 新增：target_pos分支 ---
            self.target_pos_encoder = nn.Sequential(
                nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 4)
            ).to(self.device)
            # --- 1.4 新增：elapsed_time分支 ---
            self.elapsed_time_encoder = nn.Sequential(
                nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 2)
            ).to(self.device)
            # --- 2. LNN输入自动适配 ---
            lnn_input_dim = feat_dim + 8 + 4 + 4 + 2 + 1 + action_dim
            self.lnn = LiquidNeuralNetwork(lnn_input_dim, 128, device=self.device)
            # --- 3. LNN输出归一化 ---
            self.lnn_norm = nn.LayerNorm(128).to(self.device)
            # --- 4. 多分支决策头 ---
            self.action_head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Tanh()
            ).to(self.device)
            self.q_head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
            ).to(self.device)
            self.value_head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
            ).to(self.device)
            self.optimizer = optim.Adam(self.parameters(), lr=lr or CONFIG['LR'])
        except Exception as e:
            print(f"[VisionLNNAgent] 初始化异常: {e}")
            raise

    def forward(self, state, action=None, lowdim=None, delta_pos=None, target_pos=None, elapsed_time=None, distance=None, h_prev=None):
        try:
            state = state.to(self.device).float()
            feat = self.cnn(state)
            feat = feat.view(feat.size(0), -1)
            B = state.size(0)
            if lowdim is not None:
                lowdim = lowdim.to(self.device).float().view(B, -1)
                lowdim_feat = self.lowdim_encoder(lowdim)
            else:
                lowdim_feat = torch.zeros((B, 8), device=self.device)
            if delta_pos is not None:
                delta_pos = delta_pos.to(self.device).float().view(B, 3)
                delta_pos_feat = self.delta_pos_encoder(delta_pos)
            else:
                delta_pos_feat = torch.zeros((B, 4), device=self.device)
            if target_pos is not None:
                target_pos = target_pos.to(self.device).float().view(B, 3)
                target_pos_feat = self.target_pos_encoder(target_pos)
            else:
                target_pos_feat = torch.zeros((B, 4), device=self.device)
            if elapsed_time is not None:
                elapsed_time = elapsed_time.to(self.device).float().view(B, 1)
                elapsed_time_feat = self.elapsed_time_encoder(elapsed_time)
            else:
                elapsed_time_feat = torch.zeros((B, 2), device=self.device)
            if distance is not None:
                distance = distance.to(self.device).float().view(B, 1)
            else:
                distance = torch.zeros((B, 1), device=self.device)
            if action is not None:
                action = action.to(self.device).float().view(B, -1)
            else:
                action = torch.zeros((B, self.action_dim), device=self.device, dtype=state.dtype)
            lnn_in = torch.cat([
                feat, lowdim_feat, delta_pos_feat, target_pos_feat, elapsed_time_feat, distance, action
            ], dim=1)
            lnn_out, h_new = self.lnn(lnn_in, h_prev)
            lnn_out = self.lnn_norm(lnn_out)
            action_pred = self.action_head(lnn_out)
            q_pred = self.q_head(lnn_out)
            value_pred = self.value_head(lnn_out)
            return {"action": action_pred, "q": q_pred, "value": value_pred}, h_new
        except Exception as e:
            print(f"[VisionLNNAgent] forward异常: {e}")
            raise

    def select_action(self, state, epsilon=0.1, lowdim=None, delta_pos=None, target_pos=None, elapsed_time=None, distance=None):
        try:
            action_low = getattr(self, 'action_low', -10)
            action_high = getattr(self, 'action_high', 10)
            if random.random() < epsilon:
                action = np.random.uniform(action_low, action_high, self.action_dim)
                return action
            state = state.to(self.device).float().unsqueeze(0)
            if lowdim is not None: lowdim = lowdim.to(self.device).float().view(1, -1)
            if delta_pos is not None: delta_pos = delta_pos.to(self.device).float().view(1, -1)
            if target_pos is not None: target_pos = target_pos.to(self.device).float().view(1, -1)
            if elapsed_time is not None: elapsed_time = elapsed_time.to(self.device).float().view(1, -1)
            if distance is not None: distance = distance.to(self.device).float().view(1, -1)
            with torch.no_grad():
                out, _ = self.forward(state, lowdim=lowdim, delta_pos=delta_pos, target_pos=target_pos, elapsed_time=elapsed_time, distance=distance)
                action = out["action"].squeeze(0).cpu().numpy()
            action = np.clip(action, action_low, action_high)
            return action
        except Exception as e:
            print(f"[VisionLNNAgent] select_action异常: {e}")
            raise

    def compute_loss(self, batch_states, batch_actions, batch_targets, lowdim=None, delta_pos=None, target_pos=None, elapsed_time=None, distance=None):
        try:
            batch_states = batch_states.to(self.device).float()
            batch_actions = batch_actions.to(self.device).float()
            batch_targets = batch_targets.to(self.device).float().view(-1)
            out, _ = self.forward(batch_states, batch_actions, lowdim=lowdim, delta_pos=delta_pos, target_pos=target_pos, elapsed_time=elapsed_time, distance=distance)
            q_pred = out["q"].squeeze()
            loss = nn.functional.mse_loss(q_pred, batch_targets)
            return loss
        except Exception as e:
            print(f"[VisionLNNAgent] compute_loss异常: {e}")
            raise

    def update(self, loss):
        try:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            print(f"[VisionLNNAgent] update异常: {e}")
            raise

    def save(self, path):
        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            print(f"[VisionLNNAgent] save异常: {e}")
            raise

    def load(self, path):
        try:
            self.load_state_dict(torch.load(path, map_location=self.device))
        except Exception as e:
            print(f"[VisionLNNAgent] load异常: {e}")
            raise

    def reset_state(self):
        self.lnn.reset_state(device=self.device)

    def export_onnx(self, export_path, input_shape=(1, 4, 84, 84), lowdim_shape=(1, 9)):
        """导出ONNX模型，支持跨平台部署"""
        self.eval()
        dummy_state = torch.randn(*input_shape, device=self.device)
        dummy_action = torch.randn((input_shape[0], self.action_dim), device=self.device)
        dummy_lowdim = torch.randn(*lowdim_shape, device=self.device)
        torch.onnx.export(
            self,
            (dummy_state, dummy_action, dummy_lowdim, None),
            export_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['state', 'action', 'lowdim', 'h_prev'],
            output_names=['output', 'h_new'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'action': {0: 'batch_size'},
                'lowdim': {0: 'batch_size'}
            }
        )
        print(f"[VisionLNNAgent] ONNX导出成功: {export_path}")

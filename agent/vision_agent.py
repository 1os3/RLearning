import torch
import torch.nn as nn
import torch.optim as optim
import random
from lnn.lnn_network import LiquidNeuralNetwork
from config import CONFIG

class VisionLNNAgent(nn.Module):
    """
    视觉强化学习智能体，融合CNN特征提取、LNN动态建模和多分支决策头
    新增：delta_pos（相对初始点位移）、target_pos（目标点坐标）、elapsed_time（已用时间）独立分支输入
    """
    def __init__(self, input_shape=(4, 84, 84), action_dim=3, lowdim_dim=9, device=None, lr=None):
        super().__init__()
        try:
            self.device = device or (torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu"))
            self.input_shape = input_shape
            self.action_dim = action_dim
            self.lowdim_dim = lowdim_dim
            # --- 1. 轻量级CNN特征提取 ---
            c, h, w = input_shape
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 3))
            ).to(self.device)
            cnn_feat_dim = 32 * 3 * 3
            # --- 1.1 低维状态编码（原有lowdim特征）---
            self.lowdim_encoder = nn.Sequential(
                nn.Linear(lowdim_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()
            ).to(self.device)
            # --- 1.2 新增：delta_pos分支 ---
            self.delta_pos_encoder = nn.Sequential(
                nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU()
            ).to(self.device)
            # --- 1.3 新增：target_pos分支 ---
            self.target_pos_encoder = nn.Sequential(
                nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU()
            ).to(self.device)
            # --- 1.4 新增：elapsed_time分支 ---
            self.elapsed_time_encoder = nn.Sequential(
                nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 2), nn.ReLU()
            ).to(self.device)
            # --- 2. LNN输入 ---
            # 输入维度：cnn_feat_dim + 8 + 4 + 4 + 2 + action_dim
            lnn_input_dim = cnn_feat_dim + 8 + 4 + 4 + 2 + action_dim
            from lnn.lnn_network import LiquidNeuralNetwork
            self.lnn = LiquidNeuralNetwork(input_size=lnn_input_dim, output_size=128, device=self.device)
            # --- 3. LNN输出归一化 ---
            self.lnn_norm = nn.LayerNorm(128).to(self.device)
            # --- 4. 多分支决策头 ---
            self.action_head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim)
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

    def forward(self, state, action=None, lowdim=None, delta_pos=None, target_pos=None, elapsed_time=None, h_prev=None):
        try:
            # state: (B, C, H, W)
            state = state.to(self.device).float()
            feat = self.cnn(state)
            feat = feat.view(feat.size(0), -1)
            B = state.size(0)
            # 原有lowdim
            if lowdim is not None:
                lowdim = lowdim.to(self.device).float().view(B, -1)
                lowdim_feat = self.lowdim_encoder(lowdim)
            else:
                lowdim_feat = torch.zeros((B, 8), device=self.device)
            # delta_pos分支
            if delta_pos is not None:
                delta_pos = delta_pos.to(self.device).float().view(B, 3)
                delta_pos_feat = self.delta_pos_encoder(delta_pos)
            else:
                delta_pos_feat = torch.zeros((B, 4), device=self.device)
            # target_pos分支
            if target_pos is not None:
                target_pos = target_pos.to(self.device).float().view(B, 3)
                target_pos_feat = self.target_pos_encoder(target_pos)
            else:
                target_pos_feat = torch.zeros((B, 4), device=self.device)
            # elapsed_time分支
            if elapsed_time is not None:
                elapsed_time = elapsed_time.to(self.device).float().view(B, 1)
                elapsed_time_feat = self.elapsed_time_encoder(elapsed_time)
            else:
                elapsed_time_feat = torch.zeros((B, 2), device=self.device)
            # 动作分支
            if action is not None:
                action = action.to(self.device).float().view(B, -1)
            else:
                action = torch.zeros((B, self.action_dim), device=self.device, dtype=state.dtype)
            # 拼接全部特征
            lnn_in = torch.cat([
                feat, lowdim_feat, delta_pos_feat, target_pos_feat, elapsed_time_feat, action
            ], dim=1)
            lnn_out, h_new = self.lnn(lnn_in, h_prev)
            lnn_out = self.lnn_norm(lnn_out)
            # --- 多分支输出 ---
            action_pred = self.action_head(lnn_out)
            q_pred = self.q_head(lnn_out)
            value_pred = self.value_head(lnn_out)
            return {"action": action_pred, "q": q_pred, "value": value_pred}, h_new
        except Exception as e:
            print(f"[VisionLNNAgent] forward异常: {e}")
            raise

    def select_action(self, state, epsilon=0.1):
        try:
            if random.random() < epsilon:
                action = [random.uniform(-10, 10) for _ in range(self.action_dim)]
                return action
            state = state.to(self.device).float().unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                out, _ = self.forward(state)
                action = out["action"].squeeze(0).cpu().numpy().tolist()
            action = [min(10, max(-10, a)) for a in action]
            return action
        except Exception as e:
            print(f"[VisionLNNAgent] select_action异常: {e}")
            raise

    def compute_loss(self, batch_states, batch_actions, batch_targets, lowdim=None, delta_pos=None, target_pos=None, elapsed_time=None):
        try:
            batch_states = batch_states.to(self.device).float()
            batch_actions = batch_actions.to(self.device).float()
            batch_targets = batch_targets.to(self.device).float().view(-1)
            # 新增：传递全部特征
            out, _ = self.forward(batch_states, batch_actions, lowdim=lowdim, delta_pos=delta_pos, target_pos=target_pos, elapsed_time=elapsed_time)
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

import torch
import torch.nn as nn
from .lnn_cell import LiquidNeuronCell

class LiquidNeuralNetwork(nn.Module):
    """
    多层优化版液体神经网络
    - 支持多层堆叠（depth可调）
    - 强化输出头（两层MLP+归一化+Dropout）
    - 高效批量推理
    """
    def __init__(self, input_size, output_size, neuron_num=32, depth=2, device=None, dropout=0.2):
        super().__init__()
        self.neuron_num = neuron_num
        self.depth = depth
        self.device = device or (torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu"))
        # 多层LNN Cell堆叠
        self.lnn_cells = nn.ModuleList([
            LiquidNeuronCell(input_size if i==0 else neuron_num, neuron_num, use_layernorm=True)
            for i in range(depth)
        ])
        # 输出头：两层MLP+归一化+Dropout
        self.norm = nn.LayerNorm(neuron_num)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(neuron_num, neuron_num),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neuron_num, output_size)
        )
        self.to(self.device)

    def forward(self, x, h_prev_list=None):
        try:
            x = x.to(self.device)
            h_list = []
            h_in = x
            if h_prev_list is None:
                h_prev_list = [None]*self.depth
            for i, cell in enumerate(self.lnn_cells):
                h_out = cell(h_in, h_prev_list[i])
                h_list.append(h_out)
                h_in = h_out
            out = self.norm(h_in)
            out = self.dropout(out)
            out = self.fc_out(out)
            return out, h_list
        except Exception as e:
            print(f"[LiquidNeuralNetwork] Forward error: {e}")
            raise

    def reset_state(self, device=None):
        for cell in self.lnn_cells:
            cell.reset_state(device=device)

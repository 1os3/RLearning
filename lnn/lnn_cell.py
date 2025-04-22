import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidNeuronCell(nn.Module):
    """
    优化版液体神经元单元（Liquid Time-constant Network Cell）
    - 支持批量输入
    - 增加门控机制（GLU）
    - 可选LayerNorm
    """
    def __init__(self, input_size, hidden_size, use_layernorm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layernorm = use_layernorm
        # 输入权重
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        # 门控权重（GLU）
        self.W_gate = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        # 内部递归权重
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        # 时间常数tau可训练，初始化为正
        self.tau = nn.Parameter(torch.abs(torch.randn(hidden_size)) + 1.0)
        # 偏置
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        # LayerNorm
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_size)
        # 初始状态
        self.register_buffer('h', torch.zeros(hidden_size))

    def forward(self, x, h_prev=None):
        try:
            x = x.to(self.W_in.device)
            if h_prev is None:
                h_prev = self.h.detach().to(self.W_in.device)
            else:
                h_prev = h_prev.to(self.W_in.device)
            # 计算输入和递归项
            input_term = F.linear(x, self.W_in)
            gate_term = torch.sigmoid(F.linear(x, self.W_gate))  # GLU门控
            rec_term = F.linear(h_prev, self.W_rec)
            # 液体神经元动力学：leaky integration + 门控
            dh = (-h_prev + torch.tanh(input_term * gate_term + rec_term + self.bias)) / self.tau
            h_new = h_prev + dh
            if self.use_layernorm:
                h_new = self.ln(h_new)
            return h_new
        except Exception as e:
            print(f"[LNNCell] Forward error: {e}")
            raise

    def reset_state(self, device=None):
        try:
            if device is None:
                device = self.h.device
            self.h = torch.zeros(self.hidden_size, device=device)
        except Exception as e:
            print(f"[LNNCell] Reset state error: {e}")
            raise

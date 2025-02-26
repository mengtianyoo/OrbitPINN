import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, 3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_dim)
    
    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))

class EnhancedOrbitPINN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128):
        """
        参数:
        - input_dim: 输入维度 (航天器6 + 月球6 = 12)
        - hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.tcn = TCNBlock(input_dim, hidden_dim, dilation=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 6)
        )
    
    def forward(self, x, t):
        """
        前向传播。
        参数:
        - x: 输入序列 (batch, seq_len, 12)
        - t: 预测时间步 (batch, pred_len)
        返回:
        - 预测状态 (batch, pred_len, 6)
        """
        batch_size = x.size(0)
        # TCN
        tcn_out = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)[:, -1, :]  # (batch, hidden_dim)
        
        preds = []
        current_state = x[:, -1, :6].clone()  # (batch, 6)
        for i in range(t.size(1)):
            delta_t = t[:, i:i+1]  # (batch, 1)
            inp = torch.cat([tcn_out, delta_t], dim=1)  # (batch, hidden_dim + 1)
            delta = self.fc(inp)  # (batch, 6)
            current_state = current_state + delta
            preds.append(current_state)
        
        return torch.stack(preds, dim=1)  # (batch, pred_len, 6)
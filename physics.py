import torch

def physics_loss(pred_states, moon_pos, dt=60.0, mu_earth=3.986e5, mu_moon=4.904e3):
    """
    计算物理损失，基于地球和月球引力。
    参数:
    - pred_states: 预测状态 (batch, pred_len, 6) [位置, 速度]
    - moon_pos: 月球位置 (batch, moon_len, 3)
    - dt: 时间步长 (秒)
    - mu_earth: 地球引力常数
    - mu_moon: 月球引力常数
    返回:
    - 物理损失值 (标量)
    """
    pos = pred_states[..., :3]  # (batch, pred_len, 3)
    vel = pred_states[..., 3:]  # (batch, pred_len, 3)

    # 数值微分计算加速度
    acc = (vel[:, 1:] - vel[:, :-1]) / dt  # (batch, pred_len-1, 3)
    
    # 地球引力（含J2）
    r_earth = torch.norm(pos[:, :-1], dim=-1, keepdim=True)
    earth_gravity = -mu_earth * pos[:, :-1] / (r_earth**3 + 1e-6)

    
    # 月球引力 - 现在moon_pos和pos长度一致，我们可以放心地使用
    r_moon = pos[:, :-1] - moon_pos[:, :-1]
    #moon_gravity = -mu_moon * r_moon / (torch.norm(r_moon, dim=-1, keepdim=True)**3 + 1e-6)
    moon_gravity = -mu_moon * (pos[:, :-1] - moon_pos[:, :pos.size(1)-1, :]) / (r_moon**3 + 1e-6)
    # 总加速度
    total_acc = earth_gravity + moon_gravity
    
    # 残差
    residual = acc - total_acc
    return torch.mean(residual**2)
import torch

def gravity_J2(pos, mu=3.986e5, R_earth=6378.137, J2=1.0826e-3):
    """
    计算地球引力加速度，包含J2摄动。
    参数:
    - pos: 位置向量 (batch, pred_len, 3)
    - mu: 地球引力常数 (km³/s²)
    - R_earth: 地球半径 (km)
    - J2: 地球J2摄动系数
    返回:
    - 加速度向量 (batch, pred_len, 3)
    """
    r = torch.norm(pos, dim=-1, keepdim=True)
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    factor = 1.5 * J2 * (R_earth / r)**2 * (5 * (z / r)**2 - 3)
    ax = -mu * x / r**3 * (1 + factor)
    ay = -mu * y / r**3 * (1 + factor)
    az = -mu * z / r**3 * (1 + 1.5 * J2 * (R_earth / r)**2 * (5 * (z / r)**2 - 1))
    return torch.stack([ax, ay, az], dim=-1)

def physics_loss(pred_states, moon_pos, dt=60.0, mu_earth=3.986e5, mu_moon=4.904e3):
    """
    计算物理损失，基于地球和月球引力。
    参数:
    - pred_states: 预测状态 (batch, pred_len, 6) [位置, 速度]
    - moon_pos: 月球位置 (batch, pred_len, 3)
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
    earth_gravity = gravity_J2(pos[:, :-1], mu_earth)
    
    # 月球引力
    r_moon = pos[:, :-1] - moon_pos[:, :-1]
    moon_gravity = -mu_moon * r_moon / (torch.norm(r_moon, dim=-1, keepdim=True)**3 + 1e-6)
    
    # 总加速度
    total_acc = earth_gravity + moon_gravity
    
    # 残差
    residual = acc - total_acc
    return torch.mean(residual**2)
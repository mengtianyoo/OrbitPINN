import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from physics import physics_loss

def train_model(model, X_train, y_train, moon_pos_train, epochs=1000, batch_size=32):
    """
    训练模型。
    参数:
    - model: 待训练的模型
    - X_train: 输入序列 (样本数, seq_len, 12)
    - y_train: 目标序列 (样本数, pred_len, 6)
    - moon_pos_train: 月球位置 (样本数, pred_len, 3)
    - epochs: 训练轮数
    - batch_size: 批次大小
    返回:
    - 训练好的模型
    """
    dataset = TensorDataset(X_train, y_train, moon_pos_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    log_var_data = nn.Parameter(torch.tensor(0.0))  # 数据损失权重
    log_var_phys = nn.Parameter(torch.tensor(0.0))  # 物理损失权重
    
    dt = 60.0  # 时间步长 (秒)
    t_pred = torch.arange(1, y_train.size(1) + 1).float() * dt
    t_pred = t_pred.unsqueeze(0).repeat(X_train.size(0), 1)  # (样本数, pred_len)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch, moon_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch, t_pred[:X_batch.size(0)])
            
            # 数据损失
            data_loss = nn.MSELoss()(pred, y_batch)
            # 物理损失
            phys_loss = physics_loss(pred, moon_batch, dt)
            
            # 自适应损失加权
            loss = 0.5 * (torch.exp(-log_var_data) * data_loss + log_var_data) + \
                   0.5 * (torch.exp(-log_var_phys) * phys_loss + log_var_phys)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss={total_loss/len(loader):.4f}")
    return model
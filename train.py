import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from physics import physics_loss  # 确保physics_loss已正确定义

def train_model(model, X_train, y_train, moon_pos_train, epochs=1000, batch_size=32):
    """
    训练模型，带完整日志和进度条
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
    # 数据准备
    dataset = TensorDataset(X_train, y_train, moon_pos_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器与自适应权重
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    log_var_data = nn.Parameter(torch.tensor(0.0))  # 数据损失的对数方差
    log_var_phys = nn.Parameter(torch.tensor(0.0))  # 物理损失的对数方差
    
    # 训练循环
    best_loss = float('inf')
    start_time = time.time()
    
    # 外层进度条（总epoch数）
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        total_data_loss = 0.0
        total_phys_loss = 0.0
        total_loss = 0.0
        
        # 内层进度条（batch迭代）
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (X_batch, y_batch, moon_batch) in enumerate(batch_pbar):
            optimizer.zero_grad()
            
            # 动态生成时间步（确保与batch_size匹配）
            batch_size = X_batch.size(0)
            dt = 60.0  # 时间步长 (秒)
            t_pred = torch.arange(1, y_batch.size(1)+1).float().unsqueeze(0) * dt
            t_pred = t_pred.repeat(batch_size, 1).to(X_batch.device)
            
            # 前向传播
            pred = model(X_batch, t_pred)
            
            # 损失计算
            data_loss = nn.MSELoss()(pred, y_batch)
            phys_loss = physics_loss(pred, moon_batch, dt)
            
            # 自适应加权 (Kendall's uncertainty weighting)
            loss = 0.5*(torch.exp(-log_var_data)*data_loss + log_var_data) + \
                   0.5*(torch.exp(-log_var_phys)*phys_loss + log_var_phys)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 累计损失
            total_data_loss += data_loss.item()
            total_phys_loss += phys_loss.item()
            total_loss += loss.item()
            
            # 更新batch进度条描述
            batch_pbar.set_postfix({
                'Data Loss': f"{data_loss.item():.4f}",
                'Phys Loss': f"{phys_loss.item():.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # 计算epoch平均损失
        num_batches = len(loader)
        avg_data_loss = total_data_loss / num_batches
        avg_phys_loss = total_phys_loss / num_batches
        avg_total_loss = total_loss / num_batches
        
        # 保存最佳模型
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 更新epoch进度条信息
        epoch_pbar.set_postfix({
            'Total Loss': f"{avg_total_loss:.4f}",
            'Data Loss': f"{avg_data_loss:.4f}",
            'Phys Loss': f"{avg_phys_loss:.4f}",
            'W_data': f"{torch.exp(-log_var_data).item():.2e}",
            'W_phys': f"{torch.exp(-log_var_phys).item():.2e}",
            'Time': f"{(time.time()-start_time)/60:.1f}min"
        })
        
        # 每10个epoch打印详细日志
        if (epoch+1) % 10 == 0:
            log_msg = (
                f"\nEpoch {epoch+1}/{epochs} | "
                f"Time: {(time.time()-start_time)/60:.1f}min\n"
                f"Total Loss: {avg_total_loss:.4f} | "
                f"Data Loss: {avg_data_loss:.4f} | "
                f"Phys Loss: {avg_phys_loss:.4f}\n"
                f"Data Weight: {torch.exp(-log_var_data).item():.2e} | "
                f"Phys Weight: {torch.exp(-log_var_phys).item():.2e} | "
                f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}"
            )
            tqdm.write(log_msg)
    
    print(f"\nTraining completed! Best loss: {best_loss:.4f}")
    return model
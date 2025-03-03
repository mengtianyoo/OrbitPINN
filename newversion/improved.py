import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

# 设置随机种子以确保结果可重复
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 物理常数（单位：SI）
G = 6.67430e-11  # 引力常数 (m^3 kg^-1 s^-2)
M_EARTH = 5.972e24  # 地球质量 (kg)
M_MOON = 7.342e22  # 月球质量 (kg)

### 数据集类
class SpacecraftDataset(Dataset):
    def __init__(self, data, seq_length, pred_horizon, scaler=None, mode='train'):
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.mode = mode
        features = ['t', 's_x', 's_y', 's_z', 'v_x', 'v_y', 'v_z', 'm_x', 'm_y', 'm_z']
        if scaler is None and mode == 'train':
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(data[features].values)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(data[features].values)
        self.n_sequences = len(self.data) - seq_length - pred_horizon + 1

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length + self.pred_horizon - 1, 1:4]  # s_x, s_y, s_z
        moon_pos = self.data[idx + self.seq_length + self.pred_horizon - 1, 7:10]  # m_x, m_y, m_z
        return torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(moon_pos)

    def inverse_transform_position(self, scaled_position):
        dummy = np.zeros((scaled_position.shape[0], self.scaler.n_features_in_))
        dummy[:, 1:4] = scaled_position.cpu().numpy()
        original_data = self.scaler.inverse_transform(dummy)
        return original_data[:, 1:4]

### 任务阶段检测器
class MissionPhaseDetector:
    def __init__(self, earth_near_threshold=100000, moon_near_threshold=100000):
        self.earth_near_threshold = earth_near_threshold
        self.moon_near_threshold = moon_near_threshold

    def detect_phase(self, sc_pos, moon_pos):
        earth_dist = np.sqrt(np.sum(sc_pos**2))
        moon_dist = np.sqrt(np.sum((sc_pos - moon_pos)**2))
        if earth_dist < self.earth_near_threshold:
            return 0  # 靠近地球
        elif moon_dist < self.moon_near_threshold:
            return 2  # 靠近月球
        else:
            return 1  # 过渡阶段

### 物理约束损失函数
class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_physics=0.1, adaptive_weighting=True):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.adaptive_weighting = adaptive_weighting
        self.mse_loss = nn.MSELoss()
        self.eps = 1e-8  # 避免除以零

    def forward(self, pred_pos, true_pos, sc_prev_pos, sc_prev_vel, moon_pos, dt, mission_phase=None):
        # 数据损失
        data_loss = self.mse_loss(pred_pos, true_pos)

        # 反归一化到原始单位（假设位置单位为 km，速度为 km/s）
        pred_pos_raw = pred_pos * 1e5
        sc_prev_pos_raw = sc_prev_pos * 1e5
        sc_prev_vel_raw = sc_prev_vel * 1e3
        moon_pos_raw = moon_pos * 1e5

        # 计算引力加速度
        earth_r_vec = sc_prev_pos_raw
        moon_r_vec = sc_prev_pos_raw - moon_pos_raw
        earth_r = torch.sqrt(torch.sum(earth_r_vec**2, dim=1, keepdim=True)) + self.eps
        moon_r = torch.sqrt(torch.sum(moon_r_vec**2, dim=1, keepdim=True)) + self.eps

        earth_acc = G * M_EARTH * earth_r_vec / (earth_r**3)
        moon_acc = G * M_MOON * moon_r_vec / (moon_r**3)
        total_acc = earth_acc + moon_acc

        # 物理预测位置
        physics_pred_pos = sc_prev_pos_raw + sc_prev_vel_raw * dt + 0.5 * total_acc * (dt**2)
        physics_loss = torch.log(self.mse_loss(pred_pos_raw, physics_pred_pos) + 1)  # 对数缩放

        # 自适应权重
        if self.adaptive_weighting and mission_phase is not None:
            weights = torch.ones_like(mission_phase, dtype=torch.float32)
            weights = torch.where(mission_phase == 0, 2.0, weights)  # 地球附近
            weights = torch.where(mission_phase == 2, 1.5, weights)  # 月球附近
            weights = torch.where(mission_phase == 1, 0.5, weights)  # 过渡阶段
            physics_weight = self.lambda_physics * weights.float().mean()
        else:
            physics_weight = self.lambda_physics

        total_loss = self.lambda_data * data_loss + physics_weight * physics_loss
        return total_loss, data_loss, physics_loss

### 自注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim]))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = torch.matmul(Q, K.transpose(1, 2)) / self.scale.to(x.device)
        attention_weights = torch.softmax(attention, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

### 混合 PINN 模型
class HybridPINNModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, lstm_layers=2, fc_layers=3, dropout=0.1):
        super(HybridPINNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.attention = AttentionBlock(hidden_dim)
        fc_layers_list = []
        input_size = hidden_dim
        for i in range(fc_layers - 1):
            fc_layers_list.append(nn.Linear(input_size, hidden_dim))
            fc_layers_list.append(nn.BatchNorm1d(hidden_dim))  # 添加批归一化
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(dropout))
            input_size = hidden_dim
        fc_layers_list.append(nn.Linear(input_size, 3))
        self.fc_layers = nn.Sequential(*fc_layers_list)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        last_hidden = context[:, -1, :]
        pred_pos = self.fc_layers(last_hidden)
        last_sc_pos = x[:, -1, 1:4]
        last_sc_vel = x[:, -1, 4:7]
        return pred_pos, last_sc_pos, last_sc_vel

### 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, scheduler=None, save_dir='models', phase_detector=None, dt=60.0):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    train_losses, val_losses, data_losses, physics_losses = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss, epoch_data_loss, epoch_physics_loss = 0.0, 0.0, 0.0
        for inputs, targets, moon_positions in train_loader:
            inputs, targets, moon_positions = inputs.to(device), targets.to(device), moon_positions.to(device)
            optimizer.zero_grad()
            pred_pos, last_sc_pos, last_sc_vel = model(inputs)
            if phase_detector:
                phases = [phase_detector.detect_phase(last_sc_pos[i].detach().cpu().numpy(), moon_positions[i].detach().cpu().numpy()) for i in range(inputs.size(0))]
                mission_phase = torch.tensor(phases, device=device)
            else:
                mission_phase = None
            loss, data_loss, physics_loss = criterion(pred_pos, targets, last_sc_pos, last_sc_vel, moon_positions, dt, mission_phase)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_data_loss = epoch_data_loss / len(train_loader)
        avg_physics_loss = epoch_physics_loss / len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, moon_positions in val_loader:
                inputs, targets, moon_positions = inputs.to(device), targets.to(device), moon_positions.to(device)
                pred_pos, last_sc_pos, last_sc_vel = model(inputs)
                if phase_detector:
                    phases = [phase_detector.detect_phase(last_sc_pos[i].cpu().numpy(), moon_positions[i].cpu().numpy()) for i in range(inputs.size(0))]
                    mission_phase = torch.tensor(phases, device=device)
                else:
                    mission_phase = None
                val_loss, _, _ = criterion(pred_pos, targets, last_sc_pos, last_sc_vel, moon_positions, dt, mission_phase)
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        if scheduler:
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}_valloss_{avg_val_loss:.4f}.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': avg_val_loss}, model_path)
            print(f'Model saved at {model_path}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Data Loss: {avg_data_loss:.4f}, Physics Loss: {avg_physics_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        data_losses.append(avg_data_loss)
        physics_losses.append(avg_physics_loss)

    # 绘制损失曲线（对数刻度）
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(data_losses, label='Data Loss')
    plt.plot(physics_losses, label='Physics Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

    return train_losses, val_losses, data_losses, physics_losses


def visualize_trajectory(model, test_loader, device, scaler, save_dir='results'):
    """
    Visualize model predictions against ground truth
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
        scaler: Scaler used for normalization
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    true_positions = []
    pred_positions = []
    
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            pred_pos, _, _ = model(inputs)
            
            # Store predictions and ground truth
            true_positions.append(targets.cpu().numpy())
            pred_positions.append(pred_pos.cpu().numpy())
    
    # Convert to numpy arrays
    true_positions = np.vstack(true_positions)
    pred_positions = np.vstack(pred_positions)
    
    # Inverse transform to original scale
    dummy_true = np.zeros((true_positions.shape[0], scaler.n_features_in_))
    dummy_true[:, 1:4] = true_positions
    
    dummy_pred = np.zeros((pred_positions.shape[0], scaler.n_features_in_))
    dummy_pred[:, 1:4] = pred_positions
    
    true_positions_orig = scaler.inverse_transform(dummy_true)[:, 1:4]
    pred_positions_orig = scaler.inverse_transform(dummy_pred)[:, 1:4]
    
    # Calculate errors
    errors = np.sqrt(np.sum((true_positions_orig - pred_positions_orig)**2, axis=1))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    
    print(f"Mean position error: {mean_error:.2f} km")
    print(f"Standard deviation of error: {std_error:.2f} km")
    print(f"Maximum error: {max_error:.2f} km")
    
    # 3D plot of trajectory
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot true trajectory
    ax.plot(true_positions_orig[:, 0], true_positions_orig[:, 1], true_positions_orig[:, 2], 
            'b-', linewidth=2, label='Ground Truth')
    
    # Plot predicted trajectory
    ax.plot(pred_positions_orig[:, 0], pred_positions_orig[:, 1], pred_positions_orig[:, 2], 
            'r--', linewidth=2, label='Predicted')
    
    # Plot Earth (not to scale)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    earth_radius = 6371  # km
    x = earth_radius * np.cos(u) * np.sin(v)
    y = earth_radius * np.sin(u) * np.sin(v)
    z = earth_radius * np.cos(v)
    ax.plot_surface(x, y, z, color='g', alpha=0.2)
    
    # Set labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Spacecraft Trajectory Prediction')
    ax.legend()
    
    # Save figure
    plt.savefig(os.path.join(save_dir, 'trajectory_3d.png'), dpi=300)
    plt.close()
    
    # 2D projections
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # XY plane
    axs[0, 0].plot(true_positions_orig[:, 0], true_positions_orig[:, 1], 'b-', label='Ground Truth')
    axs[0, 0].plot(pred_positions_orig[:, 0], pred_positions_orig[:, 1], 'r--', label='Predicted')
    axs[0, 0].set_xlabel('X (km)')
    axs[0, 0].set_ylabel('Y (km)')
    axs[0, 0].set_title('XY Projection')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # XZ plane
    axs[0, 1].plot(true_positions_orig[:, 0], true_positions_orig[:, 2], 'b-', label='Ground Truth')
    axs[0, 1].plot(pred_positions_orig[:, 0], pred_positions_orig[:, 2], 'r--', label='Predicted')
    axs[0, 1].set_xlabel('X (km)')
    axs[0, 1].set_ylabel('Z (km)')
    axs[0, 1].set_title('XZ Projection')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # YZ plane
    axs[1, 0].plot(true_positions_orig[:, 1], true_positions_orig[:, 2], 'b-', label='Ground Truth')
    axs[1, 0].plot(pred_positions_orig[:, 1], pred_positions_orig[:, 2], 'r--', label='Predicted')
    axs[1, 0].set_xlabel('Y (km)')
    axs[1, 0].set_ylabel('Z (km)')
    axs[1, 0].set_title('YZ Projection')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Error histogram
    axs[1, 1].hist(errors, bins=30, alpha=0.7)
    axs[1, 1].axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f} km')
    axs[1, 1].set_xlabel('Error (km)')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Position Error Distribution')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_projections.png'), dpi=300)
    plt.close()
    
    # Error over time
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.axhline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f} km')
    plt.xlabel('Time Step')
    plt.ylabel('Error (km)')
    plt.title('Position Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'error_over_time.png'), dpi=300)
    plt.close()
    
    # Save error statistics
    with open(os.path.join(save_dir, 'error_stats.txt'), 'w') as f:
        f.write(f"Mean position error: {mean_error:.2f} km\n")
        f.write(f"Standard deviation of error: {std_error:.2f} km\n")
        f.write(f"Maximum error: {max_error:.2f} km\n")
    
    return true_positions_orig, pred_positions_orig, errors

def visualize_mission_phases(model, test_loader, device, scaler, phase_detector, save_dir='results'):
    """
    Visualize trajectory with color-coded mission phases
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
        scaler: Scaler used for normalization
        phase_detector: Mission phase detector
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    true_positions = []
    pred_positions = []
    moon_positions = []
    
    with torch.no_grad():
        for inputs, targets, moon_pos in test_loader:
            inputs, targets, moon_pos = inputs.to(device), targets.to(device), moon_pos.to(device)
            
            # Forward pass
            pred_pos, _, _ = model(inputs)
            
            # Store predictions and ground truth
            true_positions.append(targets.cpu().numpy())
            pred_positions.append(pred_pos.cpu().numpy())
            moon_positions.append(moon_pos.cpu().numpy())
    
    # Convert to numpy arrays
    true_positions = np.vstack(true_positions)
    pred_positions = np.vstack(pred_positions)
    moon_positions = np.vstack(moon_positions)
    
    # Inverse transform to original scale
    dummy_true = np.zeros((true_positions.shape[0], scaler.n_features_in_))
    dummy_true[:, 1:4] = true_positions
    
    dummy_pred = np.zeros((pred_positions.shape[0], scaler.n_features_in_))
    dummy_pred[:, 1:4] = pred_positions
    
    dummy_moon = np.zeros((moon_positions.shape[0], scaler.n_features_in_))
    dummy_moon[:, 7:10] = moon_positions
    
    true_positions_orig = scaler.inverse_transform(dummy_true)[:, 1:4]
    pred_positions_orig = scaler.inverse_transform(dummy_pred)[:, 1:4]
    moon_positions_orig = scaler.inverse_transform(dummy_moon)[:, 7:10]
    
    # Detect mission phases
    phases = []
    for i in range(true_positions_orig.shape[0]):
        phase = phase_detector.detect_phase(
            true_positions_orig[i],
            moon_positions_orig[i]
        )
        phases.append(phase)
    
    phases = np.array(phases)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth (not to scale)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    earth_radius = 6371  # km
    x = earth_radius * np.cos(u) * np.sin(v)
    y = earth_radius * np.sin(u) * np.sin(v)
    z = earth_radius * np.cos(v)
    ax.plot_surface(x, y, z, color='g', alpha=0.2, label='Earth')
    
    # Moon position (average)
    mean_moon_pos = np.mean(moon_positions_orig, axis=0)
    ax.scatter(mean_moon_pos[0], mean_moon_pos[1], mean_moon_pos[2], 
              color='gray', s=100, alpha=0.5, label='Moon (Mean Position)')
    
    # Phase colors
    colors = ['blue', 'green', 'red']  # Earth, Transit, Moon
    phase_names = ['Near Earth', 'Transit', 'Near Moon']
    
    # Plot true trajectory by phase
    for phase in [0, 1, 2]:
        mask = phases == phase
        if np.any(mask):
            ax.plot(true_positions_orig[mask, 0], true_positions_orig[mask, 1], true_positions_orig[mask, 2],
                   color=colors[phase], linewidth=2, label=f'True - {phase_names[phase]}')
    
    # Plot predicted trajectory
    ax.plot(pred_positions_orig[:, 0], pred_positions_orig[:, 1], pred_positions_orig[:, 2],
           'k--', linewidth=1.5, label='Predicted')
    
    # Set labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Spacecraft Trajectory with Mission Phases')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Save figure
    plt.savefig(os.path.join(save_dir, 'trajectory_phases_3d.png'), dpi=300)
    plt.close()
    
    # Calculate errors by phase
    errors = np.sqrt(np.sum((true_positions_orig - pred_positions_orig)**2, axis=1))
    
    phase_errors = {}
    for phase in [0, 1, 2]:
        mask = phases == phase
        if np.any(mask):
            phase_errors[phase] = {
                'mean': np.mean(errors[mask]),
                'std': np.std(errors[mask]),
                'max': np.max(errors[mask]),
                'count': np.sum(mask)
            }
    
    # Plot errors by phase
    plt.figure(figsize=(12, 6))
    for phase in [0, 1, 2]:
        if phase in phase_errors:
            mask = phases == phase
            plt.scatter(np.arange(len(errors))[mask], errors[mask], 
                      color=colors[phase], alpha=0.7, label=f'{phase_names[phase]}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Error (km)')
    plt.title('Position Error by Mission Phase')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'error_by_phase.png'), dpi=300)
    plt.close()
    
    # Box plot of errors by phase
    error_data = []
    labels = []
    for phase in [0, 1, 2]:
        if phase in phase_errors:
            mask = phases == phase
            error_data.append(errors[mask])
            labels.append(f'{phase_names[phase]}\n(n={phase_errors[phase]["count"]})')
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(error_data, labels=labels)
    plt.ylabel('Error (km)')
    plt.title('Distribution of Position Errors by Mission Phase')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'error_boxplot_by_phase.png'), dpi=300)
    plt.close()
    
    # Save error statistics by phase
    with open(os.path.join(save_dir, 'phase_error_stats.txt'), 'w') as f:
        for phase in [0, 1, 2]:
            if phase in phase_errors:
                f.write(f"Phase: {phase_names[phase]}\n")
                f.write(f"  Number of points: {phase_errors[phase]['count']}\n")
                f.write(f"  Mean error: {phase_errors[phase]['mean']:.2f} km\n")
                f.write(f"  Standard deviation: {phase_errors[phase]['std']:.2f} km\n")
                f.write(f"  Maximum error: {phase_errors[phase]['max']:.2f} km\n\n")
    
    return phases, phase_errors

### 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # 加载数据（需要替换为你的实际数据）
    data = pd.read_csv('processed_data\processed_data.csv')  # 请替换为实际数据文件路径
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=RANDOM_SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=RANDOM_SEED)

    seq_length = 60
    pred_horizon = 10
    train_dataset = SpacecraftDataset(train_data, seq_length, pred_horizon, mode='train')
    scaler = train_dataset.scaler
    val_dataset = SpacecraftDataset(val_data, seq_length, pred_horizon, scaler=scaler, mode='val')
    test_dataset = SpacecraftDataset(test_data, seq_length, pred_horizon, scaler=scaler, mode='test')

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    phase_detector = MissionPhaseDetector(earth_near_threshold=50000, moon_near_threshold=50000)
    model = HybridPINNModel(input_dim=10, hidden_dim=128, lstm_layers=2, fc_layers=3, dropout=0.2).to(device)
    criterion = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=0.1, adaptive_weighting=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)

    print("Starting model training...")
    train_losses, val_losses, data_losses, physics_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=50, scheduler=scheduler, save_dir=os.path.join(run_dir, 'models'),
        phase_detector=phase_detector, dt=60.0
    )

# Load best model
    best_model_path = None
    best_val_loss = float('inf')
    for filename in os.listdir(os.path.join(run_dir, 'models')):
        if filename.endswith('.pt'):
            val_loss = float(filename.split('_')[-1].replace('.pt', ''))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(run_dir, 'models', filename)
    
    print(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate and visualize on test set
    print("Evaluating model on test set...")
    true_pos, pred_pos, errors = visualize_trajectory(
        model, test_loader, device, scaler, save_dir=os.path.join(run_dir, 'results')
    )
    
    # Visualize mission phases
    print("Generating mission phase visualizations...")
    phases, phase_errors = visualize_mission_phases(
        model, test_loader, device, scaler, phase_detector, save_dir=os.path.join(run_dir, 'results')
    )
    
    print("Completed!")


if __name__ == "__main__":
    main()
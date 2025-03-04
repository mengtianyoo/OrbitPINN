import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import math
import numpy as np
import pandas as pd

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Constants
G = 6.67430e-20  # Gravitational constant (km^3 kg^-1 s^-2)
M_EARTH = 5.972e24  # Earth mass (kg)
M_MOON = 7.342e22  # Moon mass (kg)

class SpacecraftDataset(Dataset):
    def __init__(self, data, seq_length, pred_horizon, scaler=None, mode='train'):
        """
        Dataset for spacecraft trajectory data
        
        Args:
            data: Pandas DataFrame containing spacecraft and moon positions and velocities
            seq_length: Number of time steps to use as input sequence
            pred_horizon: Number of time steps to predict into the future
            scaler: Scaler object for feature normalization
            mode: 'train', 'val', or 'test'
        """
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.mode = mode
        
        # Extract features
        features = [
            't',                # Time
            's_x', 's_y', 's_z',  # Spacecraft position
            'v_x', 'v_y', 'v_z',  # Spacecraft velocity
            'm_x', 'm_y', 'm_z'   # Moon position
        ]
        
        # Prepare data
        if scaler is None and mode == 'train':
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(data[features].values)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(data[features].values)
        
        # Calculate number of sequences
        self.n_sequences = len(self.data) - seq_length - pred_horizon + 1
        
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Input sequence
        X = self.data[idx:idx+self.seq_length]
        
        # Target: spacecraft position in future time step
        y = self.data[idx+self.seq_length+self.pred_horizon-1, 1:4]  # s_x, s_y, s_z
        
        # For physics-informed model, we also need moon position at prediction time
        moon_pos = self.data[idx+self.seq_length+self.pred_horizon-1, 7:10]  # m_x, m_y, m_z
        
        return torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(moon_pos)
    
    def inverse_transform_position(self, scaled_position):
        """Convert scaled position back to original units"""
        # Create a dummy array with zeros except for the position columns
        dummy = np.zeros((scaled_position.shape[0], self.scaler.n_features_in_))
        dummy[:, 1:4] = scaled_position.cpu().numpy()  # Position is at indices 1,2,3
        
        # Inverse transform and extract only the position columns
        original_data = self.scaler.inverse_transform(dummy)
        return original_data[:, 1:4]

def load_and_preprocess_data(processed_file):
    """
    Load and preprocess spacecraft and moon trajectory data
    
    Args:
        spacecraft_file: CSV file containing spacecraft trajectory
        moon_file: CSV file containing moon trajectory
    
    Returns:
        Pandas DataFrame with combined data
    """
    merged_data = pd.read_csv(processed_file)
    return merged_data

class MissionPhaseDetector:
    """
    Detect mission phases based on distances to Earth and Moon
    """
    def __init__(self, earth_near_threshold=100000, moon_near_threshold=100000):
        self.earth_near_threshold = earth_near_threshold
        self.moon_near_threshold = moon_near_threshold
    
    def detect_phase(self, sc_pos, moon_pos):
        """
        Detect mission phase based on spacecraft and moon positions
        
        Args:
            sc_pos: Spacecraft position [x, y, z]
            moon_pos: Moon position [x, y, z]
            
        Returns:
            phase: 0 (near Earth), 1 (transit), 2 (near Moon)
        """
        # Calculate distances
        earth_dist = np.sqrt(np.sum(sc_pos**2))
        moon_dist = np.sqrt(np.sum((sc_pos - moon_pos)**2))
        
        # Determine phase
        if earth_dist < self.earth_near_threshold:
            return 0  # Near Earth
        elif moon_dist < self.moon_near_threshold:
            return 2  # Near Moon
        else:
            return 1  # Transit between Earth and Moon


class AttentionBlock(nn.Module):
    """
    Self-attention mechanism for time series data
    """
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim]))
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        Q = self.query(x)  # [batch, seq_len, input_dim]
        K = self.key(x)    # [batch, seq_len, input_dim]
        V = self.value(x)  # [batch, seq_len, input_dim]
        
        # Compute attention scores
        attention = torch.matmul(Q, K.transpose(1, 2)) / self.scale.to(x.device)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class SpacecraftOrbitalModel(nn.Module):
    """
    Pure machine learning model for spacecraft orbit prediction
    Combines LSTM, attention, and Transformer-like components
    """
    def __init__(self, input_dim=10, hidden_dim=128, lstm_layers=2, num_heads=4, ff_dim=256, dropout=0.1):
        super(SpacecraftOrbitalModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Bidirectional LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # Divide by 2 because of bidirectional
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention_heads = nn.ModuleList([
            AttentionBlock(hidden_dim) for _ in range(num_heads)
        ])
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Position-wise feed-forward layers
        self.pos_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Final prediction layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3 outputs for (x, y, z) position
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            pred_pos: Predicted spacecraft position [batch, 3]
            last_sc_pos: Last known spacecraft position [batch, 3]
            last_sc_vel: Last known spacecraft velocity [batch, 3]
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract the last known position and velocity for returning
        last_sc_pos = x[:, -1, 1:4]  # [batch, 3]
        last_sc_vel = x[:, -1, 4:7]  # [batch, 3]
        
        # 1. Process sequence with LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        
        # 2. Apply multi-head attention (focusing on the time dimension)
        attention_outputs = []
        for attention_head in self.attention_heads:
            att_out, _ = attention_head(lstm_out)
            attention_outputs.append(att_out)
        
        # Concatenate attention heads and reshape
        multi_head_out = torch.cat([head[:, -1:, :] for head in attention_outputs], dim=1)  # [batch, num_heads, hidden_dim]
        multi_head_out = multi_head_out.view(batch_size, -1, self.hidden_dim)  # [batch, num_heads, hidden_dim]
        
        # 3. Apply attention across the multi-head dimension
        temporal_att = F.softmax(torch.sum(multi_head_out, dim=2, keepdim=True), dim=1)  # [batch, num_heads, 1]
        context = torch.sum(multi_head_out * temporal_att, dim=1)  # [batch, hidden_dim]
        
        # 4. Apply layer normalization and residual connection
        norm_context = self.layer_norm1(context)
        
        # 5. Apply position-wise FFN
        ffn_out = self.pos_ffn(norm_context)
        ffn_out = self.layer_norm2(ffn_out + norm_context)  # Residual connection
        
        # 6. Final prediction layers
        pred_pos = self.output_layers(ffn_out)  # [batch, 3]
        
        return pred_pos, last_sc_pos, last_sc_vel

def train_model(model, train_loader, val_loader, device, 
                num_epochs=50, save_dir='models', 
                phase_detector=None):
    """
    Train the model
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        num_epochs: Number of epochs to train
        save_dir: Directory to save model checkpoints
        phase_detector: Mission phase detector
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.optim as optim
    
    # Create directory for model checkpoints
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            pred_pos, _, _ = model(inputs)
            
            # Compute loss
            loss = criterion(pred_pos, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                pred_pos, _, _ = model(inputs)
                
                # Compute validation loss
                val_loss = criterion(pred_pos, targets)
                
                epoch_val_loss += val_loss.item()
        
        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}_valloss_{avg_val_loss:.4f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, model_path)
            print(f'Model saved at {model_path}')
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    return train_losses, val_losses



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

def main():
    """
    Main function to run the entire pipeline
    """
    import torch
    import os
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from datetime import datetime
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp for run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data('processed_data/processed_data.csv')
    
    # Split into train, validation, and test sets
    RANDOM_SEED = 42
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=RANDOM_SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=RANDOM_SEED)
    
    print(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    seq_length = 60  # Use 60 minutes of history
    pred_horizon = 10  # Predict 10 minutes ahead
    
    train_dataset = SpacecraftDataset(train_data, seq_length, pred_horizon, mode='train')
    scaler = train_dataset.scaler  # Get scaler from training dataset
    
    val_dataset = SpacecraftDataset(val_data, seq_length, pred_horizon, scaler=scaler, mode='val')
    test_dataset = SpacecraftDataset(test_data, seq_length, pred_horizon, scaler=scaler, mode='test')
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize mission phase detector
    phase_detector = MissionPhaseDetector(earth_near_threshold=50000, moon_near_threshold=50000)
    
    # Initialize model
    model = SpacecraftOrbitalModel(
        input_dim=10,
        hidden_dim=256,
        lstm_layers=2,
        num_heads=4,
        ff_dim=512,
        dropout=0.2
    ).to(device)
    
    # Train the model
    print("Starting model training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device, 
        num_epochs=50, save_dir=os.path.join(run_dir, 'models'),
        phase_detector=phase_detector
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

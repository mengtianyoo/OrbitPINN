import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV - fixing the typo in column names
df = pd.read_csv('orbitdata\change3.csv')
# Make sure to use different column names - there was a duplicate 'X'
# 每60个数据，采样一条 (Sample one record for every 60 data points)
df = df.iloc[::60].reset_index(drop=True)

position_data = df[['X', 'Y', 'Z']].values  # Changed 'X' to 'Y' for the second column

# Data split parameters
train_ratio = 0.7  # Use 70% for training
validation_ratio = 0.15  # Use 15% for validation (selecting starting point)
test_ratio = 0.6  # Use 15% for testing (ground truth)

# Calculate split indices
total_points = position_data.shape[0]
train_end = int(total_points * train_ratio)
validation_end = train_end + int(total_points * validation_ratio)

# Split the data
train_data = position_data[:train_end]
validation_data = position_data[train_end:validation_end]
test_data = position_data[validation_end:]

# Normalize using only the training data
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_scaled = scaler.fit_transform(train_data)
validation_data_scaled = scaler.transform(validation_data)
test_data_scaled = scaler.transform(test_data)

# Combine training and validation for the complete model training
train_val_data_scaled = np.vstack((train_data_scaled, validation_data_scaled))

# Create sequences for LSTM training
def create_sequences(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Custom Dataset for PyTorch
class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Enhanced LSTM model with more capacity
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Parameters
timesteps = 100       # Number of past points to use for prediction
hidden_size = 128     # LSTM hidden units
num_layers = 2        # Number of LSTM layers
input_size = 3        # (x, y, z)
output_size = 3       # Predict (x, y, z)

# Prepare data (using normalized values)
X, y = create_sequences(train_val_data_scaled, timesteps)
dataset = TrajectoryDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize enhanced model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model with early stopping
epochs = 150
best_loss = float('inf')
patience = 10
counter = 0
losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    batches = 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1
        
    avg_loss = epoch_loss / batches
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    # Check early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()

# Use the last sequence from validation data as our starting point
start_sequence = torch.tensor(validation_data_scaled[-timesteps:], dtype=torch.float32).unsqueeze(0).to(device)

# Predict multiple steps ahead
def predict_trajectory(model, initial_sequence, steps_ahead, scaler):
    current_sequence = initial_sequence.clone()
    predicted_trajectory = []
    
    with torch.no_grad():
        for _ in range(steps_ahead):
            # Get the next position prediction (scaled)
            pred_scaled = model(current_sequence)
            
            # Convert to numpy and inverse transform
            pred_numpy = scaler.inverse_transform(pred_scaled.cpu().numpy())
            predicted_trajectory.append(pred_numpy[0])
            
            # Update the sequence for next prediction
            # Remove oldest, add newest prediction (scaled)
            current_sequence = torch.cat((current_sequence[:, 1:, :], 
                                        pred_scaled.unsqueeze(1)), dim=1)
    
    return np.array(predicted_trajectory)

# Predict future trajectory for the same length as the test data
future_steps = len(test_data)
future_trajectory = predict_trajectory(model, start_sequence, future_steps, scaler)

# Calculate prediction error
mse = np.mean((future_trajectory - test_data) ** 2)
print(f"Mean Squared Error between prediction and actual: {mse:.6f}")

# Plot trajectories
plt.figure(figsize=(15, 10))
ax = plt.axes(projection='3d')

# Plot training and validation trajectory (blue)
ax.plot3D(train_data[:, 0], train_data[:, 1], train_data[:, 2], 
          'blue', alpha=0.5, linewidth=1, label='Training Data')
ax.plot3D(validation_data[:, 0], validation_data[:, 1], validation_data[:, 2], 
          'cyan', alpha=0.7, linewidth=2, label='Validation Data (Starting Point)')

# Plot actual test trajectory (green)
ax.plot3D(test_data[:, 0], test_data[:, 1], test_data[:, 2], 
          'green', linewidth=2, label='Actual Future Trajectory')

# Plot predicted trajectory (red)
ax.plot3D(future_trajectory[:, 0], future_trajectory[:, 1], future_trajectory[:, 2], 
          'red', linewidth=2, label='Predicted Trajectory')

# Mark the starting point for prediction
ax.scatter(validation_data[-1, 0], validation_data[-1, 1], validation_data[-1, 2], 
           color='black', s=50, marker='o', label='Prediction Start Point')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()
plt.title('Orbital Trajectory Prediction vs Actual')

# Add a text box with the prediction error
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = f"MSE: {mse:.6f}"
ax.text2D(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()
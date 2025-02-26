from data_preprocessing import load_and_preprocess_data
from model import EnhancedOrbitPINN
from train import train_model
import torch

if __name__ == "__main__":
    X, y, pos_scaler, vel_scaler = load_and_preprocess_data("artemis.csv", "moon.csv", seq_len=10, pred_len=5)
    moon_pos = X[:, -5:, 6:9]
    
    # 转换为Tensor
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    moon_tensor = torch.FloatTensor(moon_pos)

    model = EnhancedOrbitPINN(input_dim=12)

    trained_model = train_model(model, X_tensor, y_tensor, moon_tensor, epochs=1000)

    torch.save(trained_model.state_dict(), "orbit_pinn.pth")
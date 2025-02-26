import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(spacecraft_path, moon_path, seq_len=10, pred_len=5):
    """
    加载和预处理航天器和月球数据。
    参数:
    - spacecraft_path: 航天器数据文件路径 (CSV格式)
    - moon_path: 月球数据文件路径 (CSV格式)
    - seq_len: 输入序列长度
    - pred_len: 预测序列长度
    返回:
    - X: 输入序列 (样本数, seq_len, 12) 包含航天器和月球特征
    - Y: 预测序列 (样本数, pred_len, 6) 仅航天器状态
    - pos_scaler: 位置标准化器
    - vel_scaler: 速度标准化器
    """
    # 加载数据
    sc_df = pd.read_csv(spacecraft_path, skipinitialspace=True)
    moon_df = pd.read_csv(moon_path, skipinitialspace=True)
    
    assert np.allclose(sc_df['JDTDB'], moon_df['JDTDB'], atol=1e-6), "航天器与月球数据时间戳未对齐"
    
    # 提取特征 (位置 X, Y, Z 和速度 VX, VY, VZ)
    sc_features = sc_df[['X', 'Y', 'Z', 'VX', 'VY', 'VZ']].values
    moon_features = moon_df[['X', 'Y', 'Z', 'VX', 'VY', 'VZ']].values
    
    # 分别标准化位置和速度
    pos_scaler = StandardScaler()
    vel_scaler = StandardScaler()
    sc_pos = pos_scaler.fit_transform(sc_features[:, :3])  # 位置
    sc_vel = vel_scaler.fit_transform(sc_features[:, 3:])  # 速度
    moon_pos = pos_scaler.transform(moon_features[:, :3])  # 使用同一标准化器
    moon_vel = vel_scaler.transform(moon_features[:, 3:])
    
    sc_norm = np.hstack([sc_pos, sc_vel])  # (时间步数, 6)
    moon_norm = np.hstack([moon_pos, moon_vel])  # (时间步数, 6)
    
    X, Y = [], []
    for i in range(len(sc_norm) - seq_len - pred_len + 1):
        X.append(np.hstack([sc_norm[i:i+seq_len], moon_norm[i:i+seq_len]]))  # 输入: 航天器+月球
        Y.append(sc_norm[i+seq_len:i+seq_len+pred_len])  # 输出: 仅航天器
    return np.array(X), np.array(Y), pos_scaler, vel_scaler
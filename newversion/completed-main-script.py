import argparse
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


from orbit_prediction_model import (
    SpacecraftDataset, HybridPINNModel, PhysicsInformedLoss,
    MissionPhaseDetector, train_model, visualize_trajectory,
    visualize_mission_phases
)

def train_orbit_prediction_model(data_path, output_dir, config):
    """
    Train the orbit prediction model
    
    Args:
        data_path: Path to processed data file
        output_dir: Directory to save outputs
        config: Dictionary of configuration parameters
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Remove phase column if it exists (we'll detect phases during training)
    if 'phase' in data.columns:
        phase_data = data['phase'].values
        data = data.drop(columns=['phase'])
    else:
        phase_data = None
    
    # Split into train, validation, and test sets
    train_data, temp_data = train_test_split(
        data, test_size=config['test_val_size'], random_state=config['random_seed']
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=config['random_seed']
    )
    
    print(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = SpacecraftDataset(
        train_data, 
        seq_length=config['seq_length'], 
        pred_horizon=config['pred_horizon'], 
        mode='train'
    )
    scaler = train_dataset.scaler  # Get scaler from training dataset
    
    val_dataset = SpacecraftDataset(
        val_data, 
        seq_length=config['seq_length'], 
        pred_horizon=config['pred_horizon'], 
        scaler=scaler, 
        mode='val'
    )
    
    test_dataset = SpacecraftDataset(
        test_data, 
        seq_length=config['seq_length'], 
        pred_horizon=config['pred_horizon'], 
        scaler=scaler, 
        mode='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Initialize mission phase detector
    phase_detector = MissionPhaseDetector(
        earth_near_threshold=config['earth_threshold'], 
        moon_near_threshold=config['moon_threshold']
    )
    
    # Initialize model
    model = HybridPINNModel(
        input_dim=10,
        hidden_dim=config['hidden_dim'], 
        lstm_layers=config['lstm_layers'], 
        fc_layers=config['fc_layers'], 
        dropout=config['dropout']
    ).to(device)
    
    # Print model summary
    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize loss function
    criterion = PhysicsInformedLoss(
        lambda_data=config['lambda_data'], 
        lambda_physics=config['lambda_physics'], 
        adaptive_weighting=config['adaptive_weighting']
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    
    # Train the model
    print("Starting model training...")
    train_losses, val_losses, data_losses, physics_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, 
        num_epochs=config['num_epochs'], 
        scheduler=scheduler, 
        save_dir=models_dir,
        phase_detector=phase_detector, 
        dt=config['time_step']
    )
    
    # Load best model
    best_model_path = None
    best_val_loss = float('inf')
    for filename in os.listdir(models_dir):
        if filename.endswith('.pt'):
            val_loss = float(filename.split('_')[-1].replace('.pt', ''))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(models_dir, filename)
    
    print(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate and visualize on test set
    print("Evaluating model on test set...")
    true_pos, pred_pos, errors = visualize_trajectory(
        model, test_loader, device, scaler, save_dir=results_dir
    )
    
    # Visualize mission phases
    print("Generating mission phase visualizations...")
    phases, phase_errors = visualize_mission_phases(
        model, test_loader, device, scaler, phase_detector, save_dir=results_dir
    )
    
    print("Training and evaluation completed!")
    return model, scaler

def predict_orbit(model_path, data_path, output_dir, config):
    """
    Use a trained model to predict orbit trajectories
    
    Args:
        model_path: Path to saved model checkpoint
        data_path: Path to test data
        output_dir: Directory to save prediction results
        config: Configuration parameters
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = HybridPINNModel(
        input_dim=10,
        hidden_dim=config['hidden_dim'], 
        lstm_layers=config['lstm_layers'], 
        fc_layers=config['fc_layers'], 
        dropout=config['dropout']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Remove phase column if it exists
    if 'phase' in data.columns:
        data = data.drop(columns=['phase'])
    
    # Create dataset (without splitting)
    # Since we're just predicting, we use the full dataset
    dataset = SpacecraftDataset(
        data, 
        seq_length=config['seq_length'], 
        pred_horizon=config['pred_horizon'], 
        mode='test'
    )
    scaler = dataset.scaler
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Initialize mission phase detector
    phase_detector = MissionPhaseDetector(
        earth_near_threshold=config['earth_threshold'], 
        moon_near_threshold=config['moon_threshold']
    )
    
    # Perform predictions
    print("Generating predictions...")
    true_pos, pred_pos, errors = visualize_trajectory(
        model, dataloader, device, scaler, save_dir=output_dir
    )
    
    # Visualize mission phases
    print("Generating mission phase visualizations...")
    phases, phase_errors = visualize_mission_phases(
        model, dataloader, device, scaler, phase_detector, save_dir=output_dir
    )
    
    print("Prediction completed!")

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate spacecraft orbit prediction model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', required=True, help='Path to processed data file')
    train_parser.add_argument('--output', default='model_output', help='Directory to save outputs')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--seq_length', type=int, default=60, help='Sequence length (minutes)')
    train_parser.add_argument('--pred_horizon', type=int, default=10, help='Prediction horizon (minutes)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Use a trained model to predict orbits')
    predict_parser.add_argument('--model', required=True, help='Path to saved model checkpoint')
    predict_parser.add_argument('--data', required=True, help='Path to data file')
    predict_parser.add_argument('--output', default='prediction_output', help='Directory to save outputs')
    predict_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    predict_parser.add_argument('--seq_length', type=int, default=60, help='Sequence length (minutes)')
    predict_parser.add_argument('--pred_horizon', type=int, default=10, help='Prediction horizon (minutes)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Set training configuration
        config = {
            'random_seed': 42,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'seq_length': args.seq_length,
            'pred_horizon': args.pred_horizon,
            'test_val_size': 0.3,
            'hidden_dim': 128,
            'lstm_layers': 2,
            'fc_layers': 3,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'lambda_data': 1.0,
            'lambda_physics': 0.1,
            'adaptive_weighting': True,
            'earth_threshold': 50000,  # km
            'moon_threshold': 50000,   # km
            'time_step': 60.0,  # 60 seconds between time steps
            'num_workers': 4,  # Number of data loader workers
        }
        
        # Train the model
        train_orbit_prediction_model(args.data, args.output, config)
        
    elif args.command == 'predict':
        # Set prediction configuration
        config = {
            'batch_size': args.batch_size,
            'seq_length': args.seq_length,
            'pred_horizon': args.pred_horizon,
            'hidden_dim': 128,
            'lstm_layers': 2,
            'fc_layers': 3,
            'dropout': 0.2,
            'earth_threshold': 50000,  # km
            'moon_threshold': 50000,   # km
            'num_workers': 4,  # Number of data loader workers
        }
        
        # Use the model for prediction
        predict_orbit(args.model, args.data, args.output, config)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

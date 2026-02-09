"""
Script to train the rectified Flow model (Reflow).
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import yaml

# Add root directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import (
    BaseFlowModel, 
    RectifiedFlowModel, 
    generate_reflow_pairs, 
    train_rectified_flow,
    iterative_reflow
)
from experiments.train_base import ImageDataset, load_config


def main():
    # Load configuration
    config = load_config()
    
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(__file__).parent.parent / config['paths']['checkpoints']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained base model
    base_model_path = checkpoint_dir / 'base_flow_final.pt'
    
    if not base_model_path.exists():
        print("Trained base model not found.")
        print("Run first: python -m experiments.train_base")
        print("\nCreating untrained base model for demo...")
        
        base_model = BaseFlowModel(
            image_size=config['data']['image_size'],
            model_channels=config['model']['channels'],
            channel_mult=config['model']['channel_mult'],
            num_res_blocks=config['model']['num_res_blocks'],
            attention_resolutions=config['model']['attention_resolutions'],
            dropout=config['model']['dropout'],
            device=device
        )
    else:
        print(f"Loading base model from: {base_model_path}")
        base_model = BaseFlowModel(
            image_size=config['data']['image_size'],
            model_channels=config['model']['channels'],
            channel_mult=config['model']['channel_mult'],
            num_res_blocks=config['model']['num_res_blocks'],
            attention_resolutions=config['model']['attention_resolutions'],
            dropout=config['model']['dropout'],
            device=device
        )
        base_model.load(str(base_model_path))
    
    # Option 1: Single Reflow
    print("\n" + "="*60)
    print("TRAINING RECTIFIED MODEL (Single Reflow)")
    print("="*60)
    
    # Create rectified model
    rect_model = RectifiedFlowModel.from_base_model(base_model)
    
    # Generate pairs using base model
    num_pairs = min(1000, config['data']['num_mock_images'] * 10)  # Adjust according to data
    x0_data, x1_data = generate_reflow_pairs(
        base_model,
        num_pairs=num_pairs,
        num_steps=config['training_base']['num_timesteps'] // 10  # Menos pasos para velocidad
    )
    
    # Train rectified model
    losses = train_rectified_flow(
        model=rect_model,
        x0_data=x0_data,
        x1_data=x1_data,
        epochs=config['training_rectified']['epochs'],
        batch_size=config['training_rectified']['batch_size'],
        lr=config['training_rectified']['learning_rate'],
        save_path=str(checkpoint_dir / 'rectified_flow_k1'),
        save_every=config['training_rectified']['save_every']
    )
    
    # Save losses
    import numpy as np
    np.save(str(checkpoint_dir / 'rectified_flow_k1_losses.npy'), losses)
    
    # Option 2: Iterative Reflow (if configured)
    num_reflow_iters = config['training_rectified']['num_reflow_iterations']
    
    if num_reflow_iters > 1:
        print("\n" + "="*60)
        print(f"TRAINING ITERATIVE REFLOW (K={num_reflow_iters})")
        print("="*60)
        
        # Load data for reference (optional)
        data_dir = Path(__file__).parent.parent / config['data']['data_dir']
        dataset = ImageDataset(str(data_dir), config['data']['image_size'])
        dataloader = DataLoader(dataset, batch_size=config['training_rectified']['batch_size'])
        
        models = iterative_reflow(
            initial_model=base_model,
            real_data_loader=dataloader,
            num_iterations=num_reflow_iters,
            epochs_per_iter=config['training_rectified']['epochs'] // num_reflow_iters,
            num_pairs=num_pairs,
            teacher_steps=100,
            lr=config['training_rectified']['learning_rate'],
            save_dir=str(checkpoint_dir)
        )
        
        print(f"\nCreated {len(models)} iteratively rectified models")
    
    print("\nReflow training completed!")
    print(f"Model saved to: {checkpoint_dir / 'rectified_flow_k1_final.pt'}")
    
    # Compare straightness
    print("\nComparing trajectory straightness...")
    x0_test = torch.randn(4, 3, config['data']['image_size'], 
                          config['data']['image_size']).to(device)
    
    # Generate x1 with base model
    with torch.no_grad():
        x1_test = base_model.sample(noise=x0_test, num_steps=100)
    
    base_straightness = base_model.velocity_net.eval()
    rect_straightness = rect_model.compute_straightness(x0_test, x1_test)
    
    print(f"Rectified model straightness: {rect_straightness:.4f}")
    print("(Lower value = straighter trajectories)")


if __name__ == "__main__":
    main()

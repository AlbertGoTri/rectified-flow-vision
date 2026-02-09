"""
Base Flow Model implementation using Flow Matching.

Flow Matching trains a model to predict the velocity field
that transforms a noise distribution into the data distribution.

References:
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Building Normalizing Flows with Stochastic Interpolants (Albergo & Vanden-Eijnden, 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple, List
from tqdm import tqdm
import os

from .unet import UNet


class BaseFlowModel(nn.Module):
    """
    Base Flow Model implementing Flow Matching with linear interpolation.
    
    The idea is to learn a velocity field v(x_t, t) such that:
    dx_t/dt = v(x_t, t)
    
    where x_0 ~ N(0, I) (noise) and x_1 ~ p_data (data).
    
    With linear interpolation: x_t = (1-t)*x_0 + t*x_1
    The target is: v = x_1 - x_0
    """
    
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        model_channels: int = 64,
        channel_mult: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.device = device
        
        # Neural network that predicts the velocity field
        self.velocity_net = UNet(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout
        )
        
        self.to(device)
    
    def get_interpolation(self, x0: torch.Tensor, x1: torch.Tensor, 
                          t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes linear interpolation and target velocity field.
        
        Args:
            x0: Noise [B, C, H, W]
            x1: Data [B, C, H, W]
            t: Time [B]
        
        Returns:
            x_t: Interpolated state [B, C, H, W]
            target: Target velocity field [B, C, H, W]
        """
        t = t.view(-1, 1, 1, 1)
        
        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        x_t = (1 - t) * x0 + t * x1
        
        # Optimal velocity field is: v = x_1 - x_0
        target = x1 - x0
        
        return x_t, target
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predicts the velocity field.
        
        Args:
            x: Current state [B, C, H, W]
            t: Time [B]
        
        Returns:
            Predicted velocity [B, C, H, W]
        """
        return self.velocity_net(x, t)
    
    def compute_loss(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Computes training loss (MSE between predicted and target velocity).
        
        Args:
            x1: Batch of real data [B, C, H, W]
        
        Returns:
            Scalar loss
        """
        batch_size = x1.shape[0]
        
        # Sample noise
        x0 = torch.randn_like(x1)
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=x1.device)
        
        # Get interpolation and target
        x_t, target = self.get_interpolation(x0, x1, t)
        
        # Predict velocity
        pred = self.forward(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(pred, target)
        
        return loss
    
    @torch.no_grad()
    def sample(self, noise: Optional[torch.Tensor] = None,
               num_steps: int = 100,
               batch_size: int = 1,
               return_trajectory: bool = False) -> torch.Tensor:
        """
        Generates samples using Euler integration.
        
        Args:
            noise: Initial noise [B, C, H, W] or None to generate it
            num_steps: Number of integration steps
            batch_size: Batch size if noise is None
            return_trajectory: If True, returns all intermediate states
        
        Returns:
            Generated samples [B, C, H, W] or list of states if return_trajectory
        """
        self.eval()
        
        if noise is None:
            noise = torch.randn(batch_size, self.in_channels, 
                               self.image_size, self.image_size,
                               device=self.device)
        
        x = noise
        dt = 1.0 / num_steps
        
        trajectory = [x] if return_trajectory else None
        
        # Euler integration
        for i in range(num_steps):
            t = torch.ones(x.shape[0], device=self.device) * (i * dt)
            
            # Predict velocity
            v = self.forward(x, t)
            
            # Euler step
            x = x + v * dt
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return trajectory
        return x
    
    @torch.no_grad()
    def sample_with_trajectory(self, noise: torch.Tensor, 
                                num_steps: int = 100,
                                save_every: int = 10) -> List[torch.Tensor]:
        """
        Generates samples while saving the trajectory every N steps.
        
        Args:
            noise: Initial noise
            num_steps: Total number of steps
            save_every: Save every N steps
        
        Returns:
            List of states in the trajectory
        """
        self.eval()
        
        x = noise
        dt = 1.0 / num_steps
        trajectory = [x.clone()]
        
        for i in range(num_steps):
            t = torch.ones(x.shape[0], device=self.device) * (i * dt)
            v = self.forward(x, t)
            x = x + v * dt
            
            if (i + 1) % save_every == 0:
                trajectory.append(x.clone())
        
        return trajectory
    
    def save(self, path: str):
        """Guarda el modelo."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'image_size': self.image_size,
                'in_channels': self.in_channels
            }
        }, path)
        print(f"Modelo guardado en: {path}")
    
    def load(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Modelo cargado desde: {path}")


def train_base_flow(
    model: BaseFlowModel,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-4,
    save_path: Optional[str] = None,
    save_every: int = 10
) -> List[float]:
    """
    Entrena el modelo flow base.
    
    Args:
        model: Modelo a entrenar
        dataloader: DataLoader con los datos
        epochs: Número de epochs
        lr: Learning rate
        save_path: Path base para guardar checkpoints
        save_every: Guardar cada N epochs
    
    Returns:
        Lista de losses por epoch
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(model.device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(x)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Guardar checkpoint
        if save_path and (epoch + 1) % save_every == 0:
            model.save(f"{save_path}_epoch{epoch+1}.pt")
    
    # Guardar modelo final
    if save_path:
        model.save(f"{save_path}_final.pt")
    
    return losses


if __name__ == "__main__":
    # Test del modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    model = BaseFlowModel(
        image_size=64,
        model_channels=64,
        device=device
    )
    
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test de sampling
    print("\nTest de sampling...")
    samples = model.sample(batch_size=4, num_steps=10)
    print(f"Shape de muestras: {samples.shape}")
    
    # Test de loss
    print("\nTest de loss...")
    dummy_data = torch.randn(4, 3, 64, 64).to(device)
    loss = model.compute_loss(dummy_data)
    print(f"Loss: {loss.item():.4f}")

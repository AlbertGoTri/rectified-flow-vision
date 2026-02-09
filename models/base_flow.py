"""
Implementación del Flow Model base usando Flow Matching.

Flow Matching entrena un modelo para predecir el campo de velocidad
que transforma una distribución de ruido en la distribución de datos.

Referencias:
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
    Flow Model base implementando Flow Matching con interpolación lineal.
    
    La idea es aprender un campo de velocidad v(x_t, t) tal que:
    dx_t/dt = v(x_t, t)
    
    donde x_0 ~ N(0, I) (ruido) y x_1 ~ p_data (datos).
    
    Con interpolación lineal: x_t = (1-t)*x_0 + t*x_1
    El target es: v = x_1 - x_0
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
        
        # Red neuronal que predice el campo de velocidad
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
        Calcula la interpolación lineal y el campo de velocidad target.
        
        Args:
            x0: Ruido [B, C, H, W]
            x1: Datos [B, C, H, W]
            t: Tiempo [B]
        
        Returns:
            x_t: Estado interpolado [B, C, H, W]
            target: Campo de velocidad target [B, C, H, W]
        """
        t = t.view(-1, 1, 1, 1)
        
        # Interpolación lineal: x_t = (1-t)*x_0 + t*x_1
        x_t = (1 - t) * x0 + t * x1
        
        # El campo de velocidad óptimo es: v = x_1 - x_0
        target = x1 - x0
        
        return x_t, target
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predice el campo de velocidad.
        
        Args:
            x: Estado actual [B, C, H, W]
            t: Tiempo [B]
        
        Returns:
            Velocidad predicha [B, C, H, W]
        """
        return self.velocity_net(x, t)
    
    def compute_loss(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida de entrenamiento (MSE entre velocidad predicha y target).
        
        Args:
            x1: Batch de datos reales [B, C, H, W]
        
        Returns:
            Loss escalar
        """
        batch_size = x1.shape[0]
        
        # Muestrear ruido
        x0 = torch.randn_like(x1)
        
        # Muestrear tiempo uniformemente
        t = torch.rand(batch_size, device=x1.device)
        
        # Obtener interpolación y target
        x_t, target = self.get_interpolation(x0, x1, t)
        
        # Predecir velocidad
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
        Genera muestras usando integración de Euler.
        
        Args:
            noise: Ruido inicial [B, C, H, W] o None para generarlo
            num_steps: Número de pasos de integración
            batch_size: Tamaño del batch si noise es None
            return_trajectory: Si True, retorna todos los estados intermedios
        
        Returns:
            Muestras generadas [B, C, H, W] o lista de estados si return_trajectory
        """
        self.eval()
        
        if noise is None:
            noise = torch.randn(batch_size, self.in_channels, 
                               self.image_size, self.image_size,
                               device=self.device)
        
        x = noise
        dt = 1.0 / num_steps
        
        trajectory = [x] if return_trajectory else None
        
        # Integración de Euler
        for i in range(num_steps):
            t = torch.ones(x.shape[0], device=self.device) * (i * dt)
            
            # Predecir velocidad
            v = self.forward(x, t)
            
            # Paso de Euler
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
        Genera muestras guardando la trayectoria cada N pasos.
        
        Args:
            noise: Ruido inicial
            num_steps: Número total de pasos
            save_every: Guardar cada N pasos
        
        Returns:
            Lista de estados en la trayectoria
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

"""
Rectified Flow (Reflow) implementation.

The Reflow technique consists of:
1. Using an already trained flow model to generate (noise, image) pairs
2. Training a new model to go directly from noise to image
3. The resulting trajectories are "straighter", allowing fewer steps

This can be applied iteratively (Reflow-K) for even straighter trajectories.

References:
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
  (Liu et al., 2022) https://arxiv.org/abs/2209.03003
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, List
from tqdm import tqdm
import os

from .base_flow import BaseFlowModel
from .unet import UNet


class RectifiedFlowModel(BaseFlowModel):
    """
    Rectified Flow Model (Reflow).
    
    Inherits from BaseFlowModel but is trained differently:
    - Instead of pairs (random noise, real data)
    - Uses pairs (noise, image_generated_by_teacher)
    
    This "straightens" the trajectories, allowing faster generation.
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
        # Call parent constructor
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            device=device
        )
        
        self.reflow_iteration = 0
    
    @staticmethod
    def from_base_model(base_model: BaseFlowModel) -> 'RectifiedFlowModel':
        """
        Creates a RectifiedFlowModel with the same architecture as the base model.
        Optionally can copy weights as initialization.
        """
        rect_model = RectifiedFlowModel(
            image_size=base_model.image_size,
            in_channels=base_model.in_channels,
            device=base_model.device
        )
        
        # Optionally initialize with base model weights
        # rect_model.load_state_dict(base_model.state_dict())
        
        return rect_model
    
    def compute_straightness(self, x0: torch.Tensor, x1: torch.Tensor, 
                             num_points: int = 10) -> float:
        """
        Computes how "straight" the learned trajectory is.
        
        A perfectly straight trajectory would have straightness = 0.
        
        Args:
            x0: Initial point (noise)
            x1: Final point (image)
            num_points: Number of points to evaluate
        
        Returns:
            Straightness metric (lower is straighter)
        """
        self.eval()
        
        # Ideal trajectory (straight line)
        ideal_direction = x1 - x0
        
        deviations = []
        
        with torch.no_grad():
            x = x0.clone()
            dt = 1.0 / num_points
            
            for i in range(num_points):
                t = torch.ones(x.shape[0], device=self.device) * (i * dt)
                
                # Predicted velocity
                v_pred = self.forward(x, t)
                
                # Ideal velocity (constant direction)
                v_ideal = ideal_direction
                
                # Deviation
                deviation = F.mse_loss(v_pred, v_ideal).item()
                deviations.append(deviation)
                
                # Update position
                x = x + v_pred * dt
        
        return np.mean(deviations)


def generate_reflow_pairs(
    teacher_model: BaseFlowModel,
    num_pairs: int,
    batch_size: int = 32,
    num_steps: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates (noise, generated_image) pairs using the teacher model.
    
    Args:
        teacher_model: Trained flow model
        num_pairs: Number of pairs to generate
        batch_size: Batch size for generation
        num_steps: Integration steps for the teacher
    
    Returns:
        (x0_all, x1_all): Tensors with all pairs
    """
    teacher_model.eval()
    
    x0_list = []
    x1_list = []
    
    num_batches = (num_pairs + batch_size - 1) // batch_size
    
    print(f"Generating {num_pairs} pairs for Reflow...")
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating pairs"):
            current_batch = min(batch_size, num_pairs - len(x0_list) * batch_size)
            
            # Generate noise
            x0 = torch.randn(current_batch, teacher_model.in_channels,
                            teacher_model.image_size, teacher_model.image_size,
                            device=teacher_model.device)
            
            # Generate corresponding image with the teacher
            x1 = teacher_model.sample(noise=x0, num_steps=num_steps)
            
            x0_list.append(x0.cpu())
            x1_list.append(x1.cpu())
    
    x0_all = torch.cat(x0_list, dim=0)[:num_pairs]
    x1_all = torch.cat(x1_list, dim=0)[:num_pairs]
    
    print(f"Generated {x0_all.shape[0]} pairs")
    
    return x0_all, x1_all


def train_rectified_flow(
    model: RectifiedFlowModel,
    x0_data: torch.Tensor,
    x1_data: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: Optional[str] = None,
    save_every: int = 10
) -> List[float]:
    """
    Trains the rectified model using pre-generated pairs.
    
    Args:
        model: Rectified model to train
        x0_data: Tensor with noise [N, C, H, W]
        x1_data: Tensor with generated images [N, C, H, W]
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        save_path: Path to save checkpoints
        save_every: Save every N epochs
    
    Returns:
        List of losses per epoch
    """
    # Create dataset and dataloader
    dataset = TensorDataset(x0_data, x1_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Reflow Epoch {epoch+1}/{epochs}")
        for x0, x1 in pbar:
            x0 = x0.to(model.device)
            x1 = x1.to(model.device)
            
            # Sample time
            t = torch.rand(x0.shape[0], device=model.device)
            
            # Interpolation and target
            x_t, target = model.get_interpolation(x0, x1, t)
            
            # Predict velocity
            pred = model.forward(x_t, t)
            
            # Loss
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Reflow Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if save_path and (epoch + 1) % save_every == 0:
            model.save(f"{save_path}_epoch{epoch+1}.pt")
    
    # Guardar modelo final
    if save_path:
        model.save(f"{save_path}_final.pt")
    
    return losses


def iterative_reflow(
    initial_model: BaseFlowModel,
    real_data_loader: DataLoader,
    num_iterations: int = 2,
    epochs_per_iter: int = 30,
    num_pairs: int = 5000,
    teacher_steps: int = 100,
    lr: float = 1e-4,
    save_dir: Optional[str] = None
) -> List[RectifiedFlowModel]:
    """
    Aplica Reflow iterativamente para obtener trayectorias cada vez más rectas.
    
    Args:
        initial_model: Modelo flow base entrenado
        real_data_loader: DataLoader con datos reales (para referencia)
        num_iterations: Número de iteraciones de Reflow (K)
        epochs_per_iter: Epochs de entrenamiento por iteración
        num_pairs: Número de pares a generar por iteración
        teacher_steps: Pasos del teacher para generar pares
        lr: Learning rate
        save_dir: Directorio para guardar modelos
    
    Returns:
        Lista de modelos rectificados (uno por iteración)
    """
    models = []
    current_teacher = initial_model
    
    for k in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"REFLOW ITERACIÓN {k+1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Crear nuevo modelo student
        student = RectifiedFlowModel.from_base_model(current_teacher)
        student.reflow_iteration = k + 1
        
        # Generar pares usando el teacher actual
        x0_data, x1_data = generate_reflow_pairs(
            current_teacher, 
            num_pairs=num_pairs,
            num_steps=teacher_steps
        )
        
        # Entrenar student
        save_path = f"{save_dir}/reflow_k{k+1}" if save_dir else None
        train_rectified_flow(
            student, x0_data, x1_data,
            epochs=epochs_per_iter,
            lr=lr,
            save_path=save_path
        )
        
        models.append(student)
        current_teacher = student  # El student se convierte en teacher
        
        # Reducir pasos del teacher progresivamente
        teacher_steps = max(teacher_steps // 2, 10)
    
    return models


if __name__ == "__main__":
    # Test del modelo rectificado
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Crear modelo base dummy
    base_model = BaseFlowModel(image_size=64, device=device)
    
    # Crear modelo rectificado
    rect_model = RectifiedFlowModel.from_base_model(base_model)
    print(f"Modelo rectificado creado con {sum(p.numel() for p in rect_model.parameters()):,} parámetros")
    
    # Test generación de pares
    print("\nTest de generación de pares...")
    x0, x1 = generate_reflow_pairs(base_model, num_pairs=10, num_steps=10)
    print(f"x0 shape: {x0.shape}, x1 shape: {x1.shape}")
    
    # Test de rectitud
    print("\nTest de métrica de rectitud...")
    x0_test = torch.randn(2, 3, 64, 64).to(device)
    x1_test = torch.randn(2, 3, 64, 64).to(device)
    straightness = rect_model.compute_straightness(x0_test, x1_test)
    print(f"Straightness (sin entrenar): {straightness:.4f}")

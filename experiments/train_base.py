"""
Script para entrenar el modelo Flow base.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import sys
import yaml
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from models import BaseFlowModel, train_base_flow


class ImageDataset(Dataset):
    """Dataset simple para cargar imágenes desde un directorio."""
    
    def __init__(self, image_dir: str, image_size: int = 64):
        self.image_dir = image_dir
        self.image_size = image_size
        
        # Listar todas las imágenes
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(list(Path(image_dir).glob(ext)))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        print(f"Dataset cargado: {len(self.image_paths)} imágenes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)


def load_config():
    """Carga la configuración."""
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Cargar configuración
    config = load_config()
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Crear directorios
    checkpoint_dir = Path(__file__).parent.parent / config['paths']['checkpoints']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    data_dir = Path(__file__).parent.parent / config['data']['data_dir']
    
    if not data_dir.exists() or len(list(data_dir.glob('*'))) == 0:
        print("No se encontraron datos. Ejecuta primero: python -m utils.download_data")
        print("Generando datos sintéticos para demo...")
        from utils.download_data import download_data
        download_data(use_online=False)
    
    dataset = ImageDataset(str(data_dir), config['data']['image_size'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training_base']['batch_size'],
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if device == 'cuda' else False
    )
    
    # Crear modelo
    model = BaseFlowModel(
        image_size=config['data']['image_size'],
        model_channels=config['model']['channels'],
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=config['model']['dropout'],
        device=device
    )
    
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Entrenar
    print("\n" + "="*60)
    print("ENTRENANDO MODELO BASE")
    print("="*60)
    
    losses = train_base_flow(
        model=model,
        dataloader=dataloader,
        epochs=config['training_base']['epochs'],
        lr=config['training_base']['learning_rate'],
        save_path=str(checkpoint_dir / 'base_flow'),
        save_every=config['training_base']['save_every']
    )
    
    # Guardar losses
    import numpy as np
    np.save(str(checkpoint_dir / 'base_flow_losses.npy'), losses)
    
    print("\nEntrenamiento completado!")
    print(f"Modelo guardado en: {checkpoint_dir / 'base_flow_final.pt'}")
    
    # Test rápido de generación
    print("\nGenerando muestras de prueba...")
    samples = model.sample(batch_size=4, num_steps=50)
    print(f"Muestras generadas: {samples.shape}")


if __name__ == "__main__":
    main()

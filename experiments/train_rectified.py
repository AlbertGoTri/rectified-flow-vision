"""
Script para entrenar el modelo Flow rectificado (Reflow).
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import yaml

# Añadir el directorio raíz al path
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
    # Cargar configuración
    config = load_config()
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Crear directorios
    checkpoint_dir = Path(__file__).parent.parent / config['paths']['checkpoints']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo base entrenado
    base_model_path = checkpoint_dir / 'base_flow_final.pt'
    
    if not base_model_path.exists():
        print("No se encontró el modelo base entrenado.")
        print("Ejecuta primero: python -m experiments.train_base")
        print("\nCreando modelo base sin entrenar para demo...")
        
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
        print(f"Cargando modelo base desde: {base_model_path}")
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
    
    # Opción 1: Single Reflow
    print("\n" + "="*60)
    print("ENTRENANDO MODELO RECTIFICADO (Single Reflow)")
    print("="*60)
    
    # Crear modelo rectificado
    rect_model = RectifiedFlowModel.from_base_model(base_model)
    
    # Generar pares usando el modelo base
    num_pairs = min(1000, config['data']['num_mock_images'] * 10)  # Ajustar según datos
    x0_data, x1_data = generate_reflow_pairs(
        base_model,
        num_pairs=num_pairs,
        num_steps=config['training_base']['num_timesteps'] // 10  # Menos pasos para velocidad
    )
    
    # Entrenar modelo rectificado
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
    
    # Guardar losses
    import numpy as np
    np.save(str(checkpoint_dir / 'rectified_flow_k1_losses.npy'), losses)
    
    # Opción 2: Iterative Reflow (si se configura)
    num_reflow_iters = config['training_rectified']['num_reflow_iterations']
    
    if num_reflow_iters > 1:
        print("\n" + "="*60)
        print(f"ENTRENANDO ITERATIVE REFLOW (K={num_reflow_iters})")
        print("="*60)
        
        # Cargar datos para referencia (opcional)
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
        
        print(f"\nCreados {len(models)} modelos rectificados iterativamente")
    
    print("\nEntrenamiento Reflow completado!")
    print(f"Modelo guardado en: {checkpoint_dir / 'rectified_flow_k1_final.pt'}")
    
    # Comparar rectitud
    print("\nComparando rectitud de trayectorias...")
    x0_test = torch.randn(4, 3, config['data']['image_size'], 
                          config['data']['image_size']).to(device)
    
    # Generar x1 con el modelo base
    with torch.no_grad():
        x1_test = base_model.sample(noise=x0_test, num_steps=100)
    
    base_straightness = base_model.velocity_net.eval()
    rect_straightness = rect_model.compute_straightness(x0_test, x1_test)
    
    print(f"Rectitud del modelo rectificado: {rect_straightness:.4f}")
    print("(Menor valor = trayectorias más rectas)")


if __name__ == "__main__":
    main()

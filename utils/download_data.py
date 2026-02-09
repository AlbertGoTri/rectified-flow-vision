"""
Script para descargar imágenes de prueba (mock data).
Usa imágenes de datasets públicos accesibles.
"""

import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tqdm import tqdm
import yaml


def load_config():
    """Carga la configuración del proyecto."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_picsum_images(save_dir: str, num_images: int = 100, size: int = 64):
    """
    Descarga imágenes aleatorias de Lorem Picsum (servicio gratuito).
    https://picsum.photos/
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Descargando {num_images} imágenes de Lorem Picsum...")
    
    successful = 0
    for i in tqdm(range(num_images), desc="Descargando"):
        try:
            # Lorem Picsum proporciona imágenes aleatorias
            url = f"https://picsum.photos/{size}/{size}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                img.save(os.path.join(save_dir, f"image_{i:04d}.png"))
                successful += 1
        except Exception as e:
            print(f"Error descargando imagen {i}: {e}")
            continue
    
    print(f"Descargadas {successful}/{num_images} imágenes exitosamente")
    return successful


def generate_synthetic_images(save_dir: str, num_images: int = 100, size: int = 64):
    """
    Genera imágenes sintéticas si no hay conexión a internet.
    Crea patrones geométricos simples pero variados.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generando {num_images} imágenes sintéticas...")
    
    for i in tqdm(range(num_images), desc="Generando"):
        # Crear imagen con patrón aleatorio
        img_array = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Color de fondo aleatorio
        bg_color = np.random.randint(0, 256, 3)
        img_array[:, :] = bg_color
        
        # Añadir formas aleatorias
        num_shapes = np.random.randint(3, 8)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle', 'gradient'])
            color = np.random.randint(0, 256, 3)
            
            if shape_type == 'circle':
                cx, cy = np.random.randint(0, size, 2)
                radius = np.random.randint(5, size // 3)
                y, x = np.ogrid[:size, :size]
                mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
                img_array[mask] = color
                
            elif shape_type == 'rectangle':
                x1, y1 = np.random.randint(0, size - 10, 2)
                x2 = min(x1 + np.random.randint(10, size // 2), size)
                y2 = min(y1 + np.random.randint(10, size // 2), size)
                img_array[y1:y2, x1:x2] = color
                
            elif shape_type == 'gradient':
                direction = np.random.choice(['horizontal', 'vertical'])
                if direction == 'horizontal':
                    gradient = np.linspace(0, 1, size).reshape(1, -1, 1)
                else:
                    gradient = np.linspace(0, 1, size).reshape(-1, 1, 1)
                gradient = np.tile(gradient, (size if direction == 'horizontal' else 1,
                                              1 if direction == 'horizontal' else size, 3))
                img_array = (img_array * (1 - gradient) + color * gradient).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img.save(os.path.join(save_dir, f"image_{i:04d}.png"))
    
    print(f"Generadas {num_images} imágenes sintéticas")
    return num_images


def download_data(use_online: bool = True):
    """
    Función principal para descargar/generar datos.
    
    Args:
        use_online: Si True, intenta descargar de internet. Si False, genera sintéticas.
    """
    config = load_config()
    
    save_dir = os.path.join(
        os.path.dirname(__file__), '..', 
        config['data']['data_dir']
    )
    num_images = config['data']['num_mock_images']
    size = config['data']['image_size']
    
    if use_online:
        try:
            # Intentar descargar de internet
            downloaded = download_picsum_images(save_dir, num_images, size)
            if downloaded < num_images // 2:
                print("Pocas imágenes descargadas, complementando con sintéticas...")
                generate_synthetic_images(save_dir, num_images - downloaded, size)
        except Exception as e:
            print(f"Error con descarga online: {e}")
            print("Generando imágenes sintéticas como alternativa...")
            generate_synthetic_images(save_dir, num_images, size)
    else:
        generate_synthetic_images(save_dir, num_images, size)
    
    print(f"\nDatos guardados en: {save_dir}")
    print(f"Total imágenes: {len(os.listdir(save_dir))}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', 
                        help='Generar imágenes sintéticas sin conexión')
    args = parser.parse_args()
    
    download_data(use_online=not args.offline)

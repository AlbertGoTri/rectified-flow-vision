"""
Metrics to evaluate the quality of generated images.
Includes FID, LPIPS, SSIM and speed metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import time
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


class MetricsCalculator:
    """Metrics calculator for generative model evaluation."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self._lpips_model = None
        self._inception_model = None
    
    @property
    def lpips_model(self):
        """Lazy loading of LPIPS model."""
        if self._lpips_model is None:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self._lpips_model.eval()
            except ImportError:
                print("LPIPS not available. Install with: pip install lpips")
                return None
        return self._lpips_model
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Computes SSIM between two images.
        
        Args:
            img1, img2: Images as numpy arrays [H, W, C] in range [0, 255]
        
        Returns:
            SSIM value (higher is better, maximum 1.0)
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same size")
        
        # Convert to grayscale if necessary for SSIM
        if len(img1.shape) == 3:
            return ssim(img1, img2, channel_axis=2, data_range=255)
        return ssim(img1, img2, data_range=255)
    
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Computes LPIPS between two images.
        
        Args:
            img1, img2: Tensors [B, C, H, W] in range [-1, 1]
        
        Returns:
            LPIPS value (lower is better)
        """
        if self.lpips_model is None:
            return float('nan')
        
        with torch.no_grad():
            return self.lpips_model(img1.to(self.device), 
                                    img2.to(self.device)).mean().item()
    
    def compute_fid_statistics(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes statistics (mean and covariance) for FID.
        Uses a simplified version without Inception for speed.
        
        Args:
            images: Tensor [N, C, H, W]
        
        Returns:
            (mu, sigma): Mean and covariance of features
        """
        # Simplified version: use image features directly
        # In production, use Inception v3
        images_flat = images.view(images.size(0), -1).cpu().numpy()
        mu = np.mean(images_flat, axis=0)
        sigma = np.cov(images_flat, rowvar=False)
        return mu, sigma
    
    def compute_fid(self, real_images: torch.Tensor, 
                    generated_images: torch.Tensor) -> float:
        """
        Computes simplified FID (Fréchet Inception Distance).
        
        Args:
            real_images: Tensor [N, C, H, W]
            generated_images: Tensor [M, C, H, W]
        
        Returns:
            FID value (lower is better)
        """
        mu1, sigma1 = self.compute_fid_statistics(real_images)
        mu2, sigma2 = self.compute_fid_statistics(generated_images)
        
        # Compute FID
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)
    
    def compute_generation_speed(self, 
                                  model, 
                                  num_samples: int,
                                  num_steps: int,
                                  batch_size: int = 1,
                                  num_runs: int = 5,
                                  image_size: int = 64) -> Dict[str, float]:
        """
        Measures the generation speed of the model.
        
        Args:
            model: Generation model
            num_samples: Number of samples to generate
            num_steps: Number of integration steps
            batch_size: Batch size
            num_runs: Number of runs to average
            image_size: Image size
        
        Returns:
            Dictionary with times and statistics
        """
        model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # Warmup
                if _ == 0:
                    noise = torch.randn(1, 3, image_size, image_size).to(self.device)
                    _ = model.sample(noise, num_steps=num_steps)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                for i in range(0, num_samples, batch_size):
                    current_batch = min(batch_size, num_samples - i)
                    noise = torch.randn(current_batch, 3, image_size, image_size).to(self.device)
                    _ = model.sample(noise, num_steps=num_steps)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append(end_time - start_time)
        
        total_time = np.mean(times)
        time_per_image = total_time / num_samples
        images_per_second = num_samples / total_time
        
        return {
            'total_time': total_time,
            'time_per_image': time_per_image,
            'images_per_second': images_per_second,
            'time_std': np.std(times),
            'num_steps': num_steps,
            'num_samples': num_samples
        }


def benchmark_models(base_model, rectified_model, 
                     steps_list: List[int],
                     num_samples: int = 50,
                     image_size: int = 64,
                     device: str = 'cuda') -> Dict:
    """
    Compara velocidad y calidad entre modelo base y rectificado.
    
    Args:
        base_model: Modelo flow base
        rectified_model: Modelo flow rectificado
        steps_list: Lista de números de pasos a probar
        num_samples: Número de muestras para evaluar
        image_size: Tamaño de las imágenes
        device: Dispositivo de cómputo
    
    Returns:
        Diccionario con todos los resultados del benchmark
    """
    metrics_calc = MetricsCalculator(device)
    results = {
        'base_model': [],
        'rectified_model': []
    }
    
    print("\n" + "="*60)
    print("BENCHMARK: Modelo Base vs Modelo Rectificado")
    print("="*60)
    
    for num_steps in tqdm(steps_list, desc="Probando diferentes pasos"):
        # Benchmark modelo base
        base_speed = metrics_calc.compute_generation_speed(
            base_model, num_samples, num_steps, image_size=image_size
        )
        base_speed['model'] = 'base'
        results['base_model'].append(base_speed)
        
        # Benchmark modelo rectificado
        rect_speed = metrics_calc.compute_generation_speed(
            rectified_model, num_samples, num_steps, image_size=image_size
        )
        rect_speed['model'] = 'rectified'
        results['rectified_model'].append(rect_speed)
        
        print(f"\nPasos: {num_steps}")
        print(f"  Base:       {base_speed['time_per_image']*1000:.2f} ms/img")
        print(f"  Rectified:  {rect_speed['time_per_image']*1000:.2f} ms/img")
    
    return results


if __name__ == "__main__":
    # Test básico de las métricas
    calc = MetricsCalculator()
    
    # Crear imágenes dummy
    img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img2 = img1 + np.random.randint(-10, 10, (64, 64, 3))
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    ssim_val = calc.compute_ssim(img1, img2)
    print(f"SSIM entre imágenes similares: {ssim_val:.4f}")
    
    img3 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ssim_val2 = calc.compute_ssim(img1, img3)
    print(f"SSIM entre imágenes diferentes: {ssim_val2:.4f}")

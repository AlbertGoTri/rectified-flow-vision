"""
Utilidades de visualización para los experimentos de flow distillation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Optional
import os
import seaborn as sns
from PIL import Image


def setup_plot_style():
    """Configura el estilo de las gráficas."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16


def plot_speed_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Grafica la comparación de velocidad entre modelos.
    
    Args:
        results: Diccionario con resultados del benchmark
        save_path: Path para guardar la figura
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extraer datos
    base_steps = [r['num_steps'] for r in results['base_model']]
    base_times = [r['time_per_image'] * 1000 for r in results['base_model']]
    rect_times = [r['time_per_image'] * 1000 for r in results['rectified_model']]
    
    # Gráfica 1: Tiempo por imagen vs pasos
    ax1 = axes[0]
    ax1.plot(base_steps, base_times, 'o-', label='Modelo Base', linewidth=2, markersize=8)
    ax1.plot(base_steps, rect_times, 's-', label='Modelo Rectificado', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Pasos de Integración')
    ax1.set_ylabel('Tiempo por Imagen (ms)')
    ax1.set_title('Velocidad de Generación')
    ax1.legend()
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Speedup del modelo rectificado
    ax2 = axes[1]
    speedup = [b / r for b, r in zip(base_times, rect_times)]
    colors = ['green' if s > 1 else 'red' for s in speedup]
    bars = ax2.bar(range(len(base_steps)), speedup, color=colors, alpha=0.7)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax2.set_xticks(range(len(base_steps)))
    ax2.set_xticklabels(base_steps)
    ax2.set_xlabel('Número de Pasos')
    ax2.set_ylabel('Speedup (Base / Rectificado)')
    ax2.set_title('Speedup del Modelo Rectificado')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()


def plot_quality_vs_speed(results: Dict, quality_metric: str = 'fid',
                          save_path: Optional[str] = None):
    """
    Grafica la relación calidad vs velocidad.
    
    Args:
        results: Diccionario con resultados incluyendo métricas de calidad
        quality_metric: Métrica de calidad a usar ('fid', 'lpips', 'ssim')
        save_path: Path para guardar la figura
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'quality' in results:
        base_quality = results['quality']['base_model']
        rect_quality = results['quality']['rectified_model']
        
        base_speed = [r['images_per_second'] for r in results['base_model']]
        rect_speed = [r['images_per_second'] for r in results['rectified_model']]
        
        ax.scatter(base_speed, base_quality, s=100, label='Modelo Base', alpha=0.7)
        ax.scatter(rect_speed, rect_quality, s=100, label='Modelo Rectificado', alpha=0.7)
        
        ax.set_xlabel('Imágenes por Segundo')
        ax.set_ylabel(f'{quality_metric.upper()} Score')
        ax.set_title('Trade-off: Calidad vs Velocidad')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No hay datos de calidad disponibles',
                ha='center', va='center', transform=ax.transAxes)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_generated_samples(samples: torch.Tensor, 
                           title: str = "Muestras Generadas",
                           nrow: int = 4,
                           save_path: Optional[str] = None):
    """
    Visualiza una grilla de imágenes generadas.
    
    Args:
        samples: Tensor [N, C, H, W] con las imágenes
        title: Título de la figura
        nrow: Número de imágenes por fila
        save_path: Path para guardar la figura
    """
    setup_plot_style()
    
    # Convertir a numpy y reorganizar
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    
    # De [-1, 1] a [0, 1]
    samples = (samples + 1) / 2
    samples = np.clip(samples, 0, 1)
    
    n_samples = min(samples.shape[0], nrow * nrow)
    ncol = nrow
    nrow_actual = (n_samples + ncol - 1) // ncol
    
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 2, nrow_actual * 2))
    axes = np.atleast_2d(axes)
    
    for i in range(nrow_actual * ncol):
        ax = axes[i // ncol, i % ncol]
        if i < n_samples:
            img = samples[i].transpose(1, 2, 0)  # CHW -> HWC
            ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()


def plot_trajectory_comparison(base_trajectories: List[torch.Tensor],
                                rect_trajectories: List[torch.Tensor],
                                save_path: Optional[str] = None):
    """
    Compara visualmente las trayectorias del modelo base vs rectificado.
    
    Args:
        base_trajectories: Lista de estados intermedios del modelo base
        rect_trajectories: Lista de estados intermedios del modelo rectificado
        save_path: Path para guardar la figura
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, len(base_trajectories), figsize=(15, 6))
    
    for i, (base_img, rect_img) in enumerate(zip(base_trajectories, rect_trajectories)):
        # Modelo base
        if isinstance(base_img, torch.Tensor):
            base_img = base_img.cpu().numpy()
        base_img = (base_img[0].transpose(1, 2, 0) + 1) / 2
        base_img = np.clip(base_img, 0, 1)
        axes[0, i].imshow(base_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Base', fontsize=12)
        
        # Modelo rectificado
        if isinstance(rect_img, torch.Tensor):
            rect_img = rect_img.cpu().numpy()
        rect_img = (rect_img[0].transpose(1, 2, 0) + 1) / 2
        rect_img = np.clip(rect_img, 0, 1)
        axes[1, i].imshow(rect_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Rectified', fontsize=12)
        
        t = i / (len(base_trajectories) - 1)
        axes[0, i].set_title(f't={t:.2f}')
    
    plt.suptitle('Comparación de Trayectorias', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_summary_report(results: Dict, save_dir: str):
    """
    Crea un reporte resumen en texto y gráficas.
    
    Args:
        results: Diccionario con todos los resultados
        save_dir: Directorio donde guardar el reporte
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Crear reporte de texto
    report_path = os.path.join(save_dir, 'benchmark_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("REPORTE DE BENCHMARK: FLOW DISTILLATION\n")
        f.write("="*60 + "\n\n")
        
        f.write("COMPARACIÓN DE VELOCIDAD\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Pasos':<10} {'Base (ms/img)':<15} {'Rect (ms/img)':<15} {'Speedup':<10}\n")
        f.write("-"*40 + "\n")
        
        for base_r, rect_r in zip(results['base_model'], results['rectified_model']):
            steps = base_r['num_steps']
            base_time = base_r['time_per_image'] * 1000
            rect_time = rect_r['time_per_image'] * 1000
            speedup = base_time / rect_time if rect_time > 0 else 0
            f.write(f"{steps:<10} {base_time:<15.2f} {rect_time:<15.2f} {speedup:<10.2f}x\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("CONCLUSIONES\n")
        f.write("-"*40 + "\n")
        
        # Calcular speedup promedio
        speedups = []
        for base_r, rect_r in zip(results['base_model'], results['rectified_model']):
            if rect_r['time_per_image'] > 0:
                speedups.append(base_r['time_per_image'] / rect_r['time_per_image'])
        
        if speedups:
            avg_speedup = np.mean(speedups)
            f.write(f"Speedup promedio: {avg_speedup:.2f}x\n")
            f.write(f"Speedup máximo: {max(speedups):.2f}x\n")
            f.write(f"Speedup mínimo: {min(speedups):.2f}x\n")
    
    print(f"Reporte guardado en: {report_path}")
    
    # Crear gráficas
    plot_speed_comparison(results, os.path.join(save_dir, 'speed_comparison.png'))


if __name__ == "__main__":
    # Test de visualización
    setup_plot_style()
    
    # Crear datos dummy
    dummy_results = {
        'base_model': [
            {'num_steps': s, 'time_per_image': 0.01 * s, 'images_per_second': 100/s}
            for s in [1, 2, 4, 8, 16, 32, 64]
        ],
        'rectified_model': [
            {'num_steps': s, 'time_per_image': 0.008 * s, 'images_per_second': 125/s}
            for s in [1, 2, 4, 8, 16, 32, 64]
        ]
    }
    
    plot_speed_comparison(dummy_results)

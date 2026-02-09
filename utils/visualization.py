"""
Visualization utilities for flow distillation experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Optional
import os
import seaborn as sns
from PIL import Image


def setup_plot_style():
    """Configure plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16


def plot_speed_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Plots the speed comparison between models.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    base_steps = [r['num_steps'] for r in results['base_model']]
    base_times = [r['time_per_image'] * 1000 for r in results['base_model']]
    rect_times = [r['time_per_image'] * 1000 for r in results['rectified_model']]
    
    # Plot 1: Time per image vs steps
    ax1 = axes[0]
    ax1.plot(base_steps, base_times, 'o-', label='Base Model', linewidth=2, markersize=8)
    ax1.plot(base_steps, rect_times, 's-', label='Rectified Model', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Integration Steps')
    ax1.set_ylabel('Time per Image (ms)')
    ax1.set_title('Generation Speed')
    ax1.legend()
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup of rectified model
    ax2 = axes[1]
    speedup = [b / r for b, r in zip(base_times, rect_times)]
    colors = ['green' if s > 1 else 'red' for s in speedup]
    bars = ax2.bar(range(len(base_steps)), speedup, color=colors, alpha=0.7)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax2.set_xticks(range(len(base_steps)))
    ax2.set_xticklabels(base_steps)
    ax2.set_xlabel('Number of Steps')
    ax2.set_ylabel('Speedup (Base / Rectified)')
    ax2.set_title('Rectified Model Speedup')
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
    Plots the quality vs speed relationship.
    
    Args:
        results: Dictionary with results including quality metrics
        quality_metric: Quality metric to use ('fid', 'lpips', 'ssim')
        save_path: Path to save the figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'quality' in results:
        base_quality = results['quality']['base_model']
        rect_quality = results['quality']['rectified_model']
        
        base_speed = [r['images_per_second'] for r in results['base_model']]
        rect_speed = [r['images_per_second'] for r in results['rectified_model']]
        
        ax.scatter(base_speed, base_quality, s=100, label='Base Model', alpha=0.7)
        ax.scatter(rect_speed, rect_quality, s=100, label='Rectified Model', alpha=0.7)
        
        ax.set_xlabel('Images per Second')
        ax.set_ylabel(f'{quality_metric.upper()} Score')
        ax.set_title('Trade-off: Calidad vs Velocidad')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No quality data available',
                ha='center', va='center', transform=ax.transAxes)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_generated_samples(samples: torch.Tensor, 
                           title: str = "Generated Samples",
                           nrow: int = 4,
                           save_path: Optional[str] = None):
    """
    Visualizes a grid of generated images.
    
    Args:
        samples: Tensor [N, C, H, W] with images
        title: Figure title
        nrow: Number of images per row
        save_path: Path to save the figure
    """
    setup_plot_style()
    
    # Convert to numpy and reorganize
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    
    # From [-1, 1] to [0, 1]
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
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_trajectory_comparison(base_trajectories: List[torch.Tensor],
                                rect_trajectories: List[torch.Tensor],
                                save_path: Optional[str] = None):
    """
    Visually compares trajectories of base vs rectified model.
    
    Args:
        base_trajectories: List of intermediate states from base model
        rect_trajectories: List of intermediate states from rectified model
        save_path: Path to save the figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, len(base_trajectories), figsize=(15, 6))
    
    for i, (base_img, rect_img) in enumerate(zip(base_trajectories, rect_trajectories)):
        # Base model
        if isinstance(base_img, torch.Tensor):
            base_img = base_img.cpu().numpy()
        base_img = (base_img[0].transpose(1, 2, 0) + 1) / 2
        base_img = np.clip(base_img, 0, 1)
        axes[0, i].imshow(base_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Base', fontsize=12)
        
        # Rectified model
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
    
    plt.suptitle('Trajectory Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_summary_report(results: Dict, save_dir: str):
    """
    Creates a summary report with text and plots.
    
    Args:
        results: Dictionary with all results
        save_dir: Directory to save the report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create text report
    report_path = os.path.join(save_dir, 'benchmark_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BENCHMARK REPORT: FLOW DISTILLATION\n")
        f.write("="*60 + "\n\n")
        
        f.write("SPEED COMPARISON\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Steps':<10} {'Base (ms/img)':<15} {'Rect (ms/img)':<15} {'Speedup':<10}\n")
        f.write("-"*40 + "\n")
        
        for base_r, rect_r in zip(results['base_model'], results['rectified_model']):
            steps = base_r['num_steps']
            base_time = base_r['time_per_image'] * 1000
            rect_time = rect_r['time_per_image'] * 1000
            speedup = base_time / rect_time if rect_time > 0 else 0
            f.write(f"{steps:<10} {base_time:<15.2f} {rect_time:<15.2f} {speedup:<10.2f}x\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("-"*40 + "\n")
        
        # Calculate average speedup
        speedups = []
        for base_r, rect_r in zip(results['base_model'], results['rectified_model']):
            if rect_r['time_per_image'] > 0:
                speedups.append(base_r['time_per_image'] / rect_r['time_per_image'])
        
        if speedups:
            avg_speedup = np.mean(speedups)
            f.write(f"Average speedup: {avg_speedup:.2f}x\n")
            f.write(f"Maximum speedup: {max(speedups):.2f}x\n")
            f.write(f"Minimum speedup: {min(speedups):.2f}x\n")
    
    print(f"Report saved to: {report_path}")
    
    # Create plots
    plot_speed_comparison(results, os.path.join(save_dir, 'speed_comparison.png'))


if __name__ == "__main__":
    # Visualization test
    setup_plot_style()
    
    # Create dummy data
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

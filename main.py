"""
Main script to run the complete Flow Distillation pipeline.

This script:
1. Downloads/generates test data
2. Trains the base flow model
3. Trains the rectified model (Reflow)
4. Runs comparative benchmark
5. Generates visualizations and report

Usage:
    python main.py                    # Full pipeline
    python main.py --skip-training    # Benchmark only (requires pre-trained models)
    python main.py --quick            # Quick training for demo
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def load_config():
    """Load configuration."""
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def update_config_for_quick_mode(config):
    """Modify configuration for quick/demo mode."""
    config['data']['num_mock_images'] = 50
    config['training_base']['epochs'] = 5
    config['training_base']['batch_size'] = 8
    config['training_rectified']['epochs'] = 3
    config['training_rectified']['num_reflow_iterations'] = 1
    config['benchmark']['num_samples'] = 10
    config['benchmark']['steps_to_test'] = [1, 4, 16, 64]
    config['benchmark']['num_runs'] = 2
    return config


def save_config(config, path):
    """Save modified configuration."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description='Flow Distillation - Pipeline Completo')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and only run benchmark')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with fewer epochs for demo')
    parser.add_argument('--offline', action='store_true',
                        help='Use synthetic data without internet connection')
    args = parser.parse_args()
    
    # Banner
    print("="*60)
    print("   FLOW DISTILLATION - Rectified Flow Testing")
    print("="*60)
    print()
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Load and possibly modify configuration
    config = load_config()
    
    if args.quick:
        print("⚡ QUICK MODE activated - Reduced configuration for demo\n")
        config = update_config_for_quick_mode(config)
        # Save temporary config
        temp_config_path = Path(__file__).parent / 'configs' / 'config_quick.yaml'
        save_config(config, temp_config_path)
    
    # ========================================
    # STEP 1: DOWNLOAD/GENERATE DATA
    # ========================================
    if not args.skip_download:
        print("="*60)
        print("STEP 1: Preparing test data")
        print("="*60)
        
        from utils.download_data import download_data
        download_data(use_online=not args.offline)
        print()
    
    # ========================================
    # STEP 2: TRAIN BASE MODEL
    # ========================================
    if not args.skip_training:
        print("="*60)
        print("STEP 2: Training base Flow model")
        print("="*60)
        
        from experiments.train_base import main as train_base_main
        train_base_main()
        print()
        
        # ========================================
        # STEP 3: TRAIN RECTIFIED MODEL
        # ========================================
        print("="*60)
        print("STEP 3: Training rectified Flow model (Reflow)")
        print("="*60)
        
        from experiments.train_rectified import main as train_rect_main
        train_rect_main()
        print()
    
    # ========================================
    # STEP 4: COMPARATIVE BENCHMARK
    # ========================================
    print("="*60)
    print("STEP 4: Running comparative benchmark")
    print("="*60)
    
    from experiments.benchmark import main as benchmark_main
    benchmark_main()
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("   PIPELINE COMPLETED")
    print("="*60)
    
    results_dir = Path(__file__).parent / config['paths']['results']
    checkpoint_dir = Path(__file__).parent / config['paths']['checkpoints']
    
    print(f"""
Generated files:

📁 Checkpoints:
   {checkpoint_dir}/
   ├── base_flow_final.pt          (Base model)
   └── rectified_flow_k1_final.pt  (Rectified model)

📁 Results:
   {results_dir}/
   ├── benchmark_results.csv       (Numerical data)
   ├── speed_comparison.png        (Speed comparison plot)
   ├── benchmark_report.txt        (Text report)
   └── *_samples_*.png             (Generated samples)

📖 Next steps:
   1. Review images in results/ to compare visual quality
   2. Check benchmark_results.csv for detailed analysis
   3. Adjust configs/config.yaml for longer experiments
   4. Try with your own data in data/mock_images/
""")
    
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()

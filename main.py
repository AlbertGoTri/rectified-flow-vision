"""
Script principal para ejecutar el pipeline completo de Flow Distillation.

Este script:
1. Descarga/genera datos de prueba
2. Entrena el modelo flow base
3. Entrena el modelo rectificado (Reflow)
4. Ejecuta benchmark comparativo
5. Genera visualizaciones y reporte

Uso:
    python main.py                    # Pipeline completo
    python main.py --skip-training    # Solo benchmark (requiere modelos pre-entrenados)
    python main.py --quick            # Entrenamiento rápido para demo
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch

# Añadir el directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))


def load_config():
    """Carga la configuración."""
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def update_config_for_quick_mode(config):
    """Modifica la configuración para modo rápido/demo."""
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
    """Guarda la configuración modificada."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description='Flow Distillation - Pipeline Completo')
    parser.add_argument('--skip-training', action='store_true',
                        help='Saltar entrenamiento y solo hacer benchmark')
    parser.add_argument('--skip-download', action='store_true',
                        help='Saltar descarga de datos')
    parser.add_argument('--quick', action='store_true',
                        help='Modo rápido con menos epochs para demo')
    parser.add_argument('--offline', action='store_true',
                        help='Usar datos sintéticos sin conexión a internet')
    args = parser.parse_args()
    
    # Banner
    print("="*60)
    print("   FLOW DISTILLATION - Rectified Flow Testing")
    print("="*60)
    print()
    
    # Configuración
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Cargar y posiblemente modificar configuración
    config = load_config()
    
    if args.quick:
        print("⚡ MODO RÁPIDO activado - Configuración reducida para demo\n")
        config = update_config_for_quick_mode(config)
        # Guardar config temporal
        temp_config_path = Path(__file__).parent / 'configs' / 'config_quick.yaml'
        save_config(config, temp_config_path)
    
    # ========================================
    # PASO 1: DESCARGAR/GENERAR DATOS
    # ========================================
    if not args.skip_download:
        print("="*60)
        print("PASO 1: Preparando datos de prueba")
        print("="*60)
        
        from utils.download_data import download_data
        download_data(use_online=not args.offline)
        print()
    
    # ========================================
    # PASO 2: ENTRENAR MODELO BASE
    # ========================================
    if not args.skip_training:
        print("="*60)
        print("PASO 2: Entrenando modelo Flow base")
        print("="*60)
        
        from experiments.train_base import main as train_base_main
        train_base_main()
        print()
        
        # ========================================
        # PASO 3: ENTRENAR MODELO RECTIFICADO
        # ========================================
        print("="*60)
        print("PASO 3: Entrenando modelo Flow rectificado (Reflow)")
        print("="*60)
        
        from experiments.train_rectified import main as train_rect_main
        train_rect_main()
        print()
    
    # ========================================
    # PASO 4: BENCHMARK COMPARATIVO
    # ========================================
    print("="*60)
    print("PASO 4: Ejecutando benchmark comparativo")
    print("="*60)
    
    from experiments.benchmark import main as benchmark_main
    benchmark_main()
    
    # ========================================
    # RESUMEN FINAL
    # ========================================
    print("\n" + "="*60)
    print("   PIPELINE COMPLETADO")
    print("="*60)
    
    results_dir = Path(__file__).parent / config['paths']['results']
    checkpoint_dir = Path(__file__).parent / config['paths']['checkpoints']
    
    print(f"""
Archivos generados:

📁 Checkpoints:
   {checkpoint_dir}/
   ├── base_flow_final.pt          (Modelo base)
   └── rectified_flow_k1_final.pt  (Modelo rectificado)

📁 Resultados:
   {results_dir}/
   ├── benchmark_results.csv       (Datos numéricos)
   ├── speed_comparison.png        (Gráfica de velocidad)
   ├── benchmark_report.txt        (Reporte de texto)
   └── *_samples_*.png             (Muestras generadas)

📖 Próximos pasos:
   1. Revisa las imágenes en results/ para comparar calidad visual
   2. Consulta benchmark_results.csv para análisis detallado
   3. Ajusta configs/config.yaml para experimentos más largos
   4. Prueba con tus propios datos en data/mock_images/
""")
    
    print("¡Experimento completado exitosamente!")


if __name__ == "__main__":
    main()

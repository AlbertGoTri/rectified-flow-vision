# Flow Model Distillation - Rectified Flow Testing

## Descripción

Este proyecto implementa y evalúa la técnica de **Rectified Flow (Reflow)**, que consiste en aplicar un flow model sobre otro ya entrenado para "enderezar" las trayectorias de generación, permitiendo:

- **Menos pasos de inferencia** (de 100+ pasos a 1-4 pasos)
- **Mayor velocidad de generación**
- **Calidad comparable** al modelo original

## Estructura del Proyecto

```
flow_distillation/
├── data/
│   └── mock_images/          # Imágenes descargadas para testing
├── models/
│   ├── base_flow.py          # Flow model base (teacher)
│   ├── rectified_flow.py     # Flow model rectificado (student)
│   └── unet.py               # Arquitectura UNet para el modelo
├── utils/
│   ├── download_data.py      # Script para descargar imágenes
│   ├── metrics.py            # FID, LPIPS, SSIM
│   └── visualization.py      # Visualización de resultados
├── experiments/
│   ├── train_base.py         # Entrenar modelo base
│   ├── train_rectified.py    # Entrenar modelo rectificado
│   └── benchmark.py          # Comparar velocidades
├── configs/
│   └── config.yaml           # Configuración del experimento
├── results/                  # Resultados guardados
├── requirements.txt
└── main.py                   # Script principal
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso Rápido

```bash
# 1. Descargar datos de prueba
python -m utils.download_data

# 2. Ejecutar benchmark completo
python main.py

# 3. O ejecutar pasos individuales
python -m experiments.train_base
python -m experiments.train_rectified
python -m experiments.benchmark
```

## Técnica: Rectified Flow

La idea principal es:

1. **Modelo Base**: Entrena un flow model estándar que aprende a transformar ruido → imagen
2. **Reflow**: Usa el modelo base para generar pares (ruido, imagen) y entrena un nuevo modelo para ir directamente de uno a otro
3. **Resultado**: Trayectorias más rectas = menos pasos necesarios

## Referencias

- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Consistency Models](https://arxiv.org/abs/2303.01469)

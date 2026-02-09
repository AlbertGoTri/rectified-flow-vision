# Flow Model Distillation - Rectified Flow Testing

## Description

This project implements and evaluates the **Rectified Flow (Reflow)** technique, which consists of applying a flow model on top of another already trained model to "straighten" the generation trajectories, allowing:

- **Fewer inference steps** (from 100+ steps to 1-4 steps)
- **Faster generation speed**
- **Comparable quality** to the original model

## Project Structure

```
flow_distillation/
├── data/
│   └── mock_images/          # Downloaded images for testing
├── models/
│   ├── base_flow.py          # Base flow model (teacher)
│   ├── rectified_flow.py     # Rectified flow model (student)
│   └── unet.py               # UNet architecture for the model
├── utils/
│   ├── download_data.py      # Script to download images
│   ├── metrics.py            # FID, LPIPS, SSIM
│   └── visualization.py      # Results visualization
├── experiments/
│   ├── train_base.py         # Train base model
│   ├── train_rectified.py    # Train rectified model
│   └── benchmark.py          # Compare speeds
├── configs/
│   └── config.yaml           # Experiment configuration
├── results/                  # Saved results
├── requirements.txt
└── main.py                   # Main script
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Download test data
python -m utils.download_data

# 2. Run complete benchmark
python main.py

# 3. Or run individual steps
python -m experiments.train_base
python -m experiments.train_rectified
python -m experiments.benchmark
```

## Technique: Rectified Flow

The main idea is:

1. **Base Model**: Train a standard flow model that learns to transform noise → image
2. **Reflow**: Use the base model to generate (noise, image) pairs and train a new model to go directly from one to the other
3. **Result**: Straighter trajectories = fewer steps needed

## References

- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Consistency Models](https://arxiv.org/abs/2303.01469)

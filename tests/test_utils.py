"""
Unit tests for Flow Distillation (Vision).
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create a MetricsCalculator instance."""
        return MetricsCalculator(device='cpu')
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initializes correctly."""
        assert calculator.device == 'cpu'
        assert calculator._lpips_model is None  # Lazy loaded
    
    def test_ssim_identical(self, calculator):
        """Test SSIM for identical images."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = calculator.compute_ssim(img, img)
        assert score > 0.99  # Should be ~1.0
    
    def test_ssim_different(self, calculator):
        """Test SSIM for completely different images."""
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.ones((64, 64, 3), dtype=np.uint8) * 255
        score = calculator.compute_ssim(img1, img2)
        assert score < 0.5  # Should be low
    
    def test_ssim_shape_mismatch(self, calculator):
        """Test SSIM raises error for mismatched shapes."""
        img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            calculator.compute_ssim(img1, img2)
    
    def test_fid_statistics_shape(self, calculator):
        """Test FID statistics have correct shape."""
        images = torch.randn(10, 3, 64, 64)
        mu, sigma = calculator.compute_fid_statistics(images)
        
        expected_features = 3 * 64 * 64
        assert mu.shape == (expected_features,)
        assert sigma.shape == (expected_features, expected_features)
    
    def test_fid_identical_batches(self, calculator):
        """Test FID for identical image batches."""
        images = torch.randn(10, 3, 32, 32)
        fid = calculator.compute_fid(images, images)
        # FID should be very low (close to 0) for identical sets
        assert fid < 1.0
    
    def test_fid_different_batches(self, calculator):
        """Test FID for different image batches."""
        images1 = torch.randn(10, 3, 32, 32)
        images2 = torch.randn(10, 3, 32, 32) * 2 + 1
        fid = calculator.compute_fid(images1, images2)
        # FID should be non-zero for different sets
        assert fid > 0


class TestImageProcessing:
    """Tests for image processing utilities."""
    
    def test_tensor_normalization(self):
        """Test image tensor normalization."""
        # Simulate normalization [-1, 1]
        img = torch.rand(1, 3, 64, 64)  # [0, 1]
        normalized = img * 2 - 1  # [-1, 1]
        
        assert normalized.min() >= -1
        assert normalized.max() <= 1
    
    def test_tensor_denormalization(self):
        """Test image tensor denormalization."""
        # Simulate denormalization [-1, 1] -> [0, 1]
        normalized = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        denormalized = (normalized + 1) / 2
        
        assert denormalized.min() >= 0
        assert denormalized.max() <= 1


class TestModelHelpers:
    """Tests for model helper functions."""
    
    def test_interpolation_t0(self):
        """Test linear interpolation at t=0 returns x0."""
        x0 = torch.randn(2, 3, 32, 32)
        x1 = torch.randn(2, 3, 32, 32)
        t = torch.zeros(2)
        
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        assert torch.allclose(x_t, x0)
    
    def test_interpolation_t1(self):
        """Test linear interpolation at t=1 returns x1."""
        x0 = torch.randn(2, 3, 32, 32)
        x1 = torch.randn(2, 3, 32, 32)
        t = torch.ones(2)
        
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        assert torch.allclose(x_t, x1)
    
    def test_interpolation_midpoint(self):
        """Test linear interpolation at t=0.5 returns average."""
        x0 = torch.zeros(2, 3, 32, 32)
        x1 = torch.ones(2, 3, 32, 32)
        t = torch.ones(2) * 0.5
        
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        expected = torch.ones(2, 3, 32, 32) * 0.5
        assert torch.allclose(x_t, expected)
    
    def test_velocity_target(self):
        """Test velocity target computation."""
        x0 = torch.randn(2, 3, 32, 32)
        x1 = torch.randn(2, 3, 32, 32)
        
        # For linear interpolation, velocity = x1 - x0
        velocity = x1 - x0
        
        assert velocity.shape == x0.shape


class TestDataLoading:
    """Tests for data loading utilities."""
    
    def test_batch_shapes(self):
        """Test that batched data has correct shapes."""
        batch_size = 4
        image_size = 64
        channels = 3
        
        # Simulate a batch
        batch = torch.randn(batch_size, channels, image_size, image_size)
        
        assert batch.shape == (4, 3, 64, 64)
    
    def test_noise_distribution(self):
        """Test that sampled noise follows standard normal."""
        torch.manual_seed(42)
        noise = torch.randn(1000, 3, 32, 32)
        
        # Check approximate mean and std
        assert abs(noise.mean().item()) < 0.1
        assert abs(noise.std().item() - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

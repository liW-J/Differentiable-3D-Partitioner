"""
Unit tests for the Tensor interface
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partitioner3d.interfaces.tensor_interface import TensorInterface


class TestTensorInterface(unittest.TestCase):
    """Test cases for Tensor Interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = TensorInterface(
            num_partitions_x=2,
            num_partitions_y=2,
            num_partitions_z=2,
            normalize_inputs=True
        )
        
        np.random.seed(42)
        
    def test_initialization(self):
        """Test interface initialization."""
        self.assertIsNotNone(self.interface.partitioner)
        self.assertTrue(self.interface.normalize_inputs)
        
    def test_partition_from_tensors(self):
        """Test partitioning from tensor inputs."""
        positions = np.random.rand(20, 3) * 1000
        sizes = np.random.rand(20, 3) * 50
        
        assignments, results = self.interface.partition_from_tensors(
            positions=positions,
            sizes=sizes
        )
        
        self.assertEqual(len(assignments), 20)
        self.assertIn('num_cells', results)
        self.assertEqual(results['num_cells'], 20)
        
    def test_partition_with_terminals(self):
        """Test partitioning with terminal awareness."""
        positions = np.random.rand(10, 3) * 1000
        terminal_positions = np.array([[0, 500, 0], [1000, 500, 0]])
        
        assignments, results = self.interface.partition_from_tensors(
            positions=positions,
            terminal_positions=terminal_positions
        )
        
        self.assertEqual(results['num_terminals'], 2)
        
    def test_get_partition_masks(self):
        """Test partition mask generation."""
        positions = np.random.rand(20, 3) * 1000
        assignments, _ = self.interface.partition_from_tensors(positions)
        
        masks = self.interface.get_partition_masks(assignments)
        
        self.assertEqual(len(masks), 8)  # 2x2x2
        for mask in masks.values():
            self.assertEqual(len(mask), 20)
            
    def test_to_one_hot(self):
        """Test one-hot encoding conversion."""
        positions = np.random.rand(20, 3) * 1000
        assignments, _ = self.interface.partition_from_tensors(positions)
        
        one_hot = self.interface.to_one_hot(assignments)
        
        self.assertEqual(one_hot.shape, (20, 8))
        # Each row should sum to 1
        np.testing.assert_array_almost_equal(one_hot.sum(axis=1), np.ones(20))
        
    def test_get_partition_centers(self):
        """Test partition center computation."""
        positions = np.random.rand(10, 3) * 1000
        self.interface.partition_from_tensors(positions)
        
        centers = self.interface.get_partition_centers()
        
        self.assertEqual(len(centers), 8)  # 2x2x2
        self.assertEqual(centers.shape[1], 3)  # 3D centers
        
    def test_partition_batch(self):
        """Test batch partitioning."""
        batch_positions = [
            np.random.rand(10, 3) * 1000,
            np.random.rand(15, 3) * 1000,
            np.random.rand(12, 3) * 1000
        ]
        
        results = self.interface.partition_batch(batch_positions)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][1]['num_cells'], 10)
        self.assertEqual(results[1][1]['num_cells'], 15)
        self.assertEqual(results[2][1]['num_cells'], 12)
        
    def test_normalization(self):
        """Test coordinate normalization."""
        positions = np.array([[0, 0, 0], [1000, 1000, 1000]])
        
        assignments, results = self.interface.partition_from_tensors(positions)
        
        self.assertIsNotNone(results['normalization_params'])
        self.assertIn('min', results['normalization_params'])
        self.assertIn('max', results['normalization_params'])
        

if __name__ == '__main__':
    unittest.main()

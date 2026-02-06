'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2026-02-06 21:01:46
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-02-06 21:06:57
FilePath: /Differentiable-3D-Partitioner/examples/example_tensor_interface.py
Description: Demonstrates how to partition a design using tensor inputs
'''

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partitioner import TensorInterface


def example_tensor_partition():
    """Example of partitioning from tensor inputs."""

    print("=" * 70)
    print("Example: 3D Partitioning from Tensor Inputs")
    print("=" * 70)

    # Initialize Tensor interface
    # Configure partition grid: 2x2x2 = 8 partitions
    interface = TensorInterface(use_terminal_awareness=True,
                                optimization_mode="balanced",
                                normalize_inputs=True)

    # Generate synthetic cell positions
    np.random.seed(42)
    num_cells = 20

    # Random positions in 3D space
    positions = np.random.rand(num_cells, 3) * 1000

    # Optional: cell sizes
    sizes = np.random.rand(num_cells, 3) * 50 + 10

    # Optional: terminal positions (I/O pins)
    terminal_positions = np.array([
        [0, 500, 0],  # Left edge
        [1000, 500, 0],  # Right edge
        [500, 0, 0],  # Bottom edge
        [500, 1000, 0]  # Top edge
    ])

    # Optional: cell names
    cell_names = [f"cell_{i}" for i in range(num_cells)]

    # Perform partitioning
    print("\nPerforming partitioning...")
    assignments, results = interface.partition_from_tensors(
        positions=positions,
        sizes=sizes,
        terminal_positions=terminal_positions,
        cell_names=cell_names)

    # Print summary
    print("\n" + interface.get_partition_summary(results))

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    example_tensor_partition()

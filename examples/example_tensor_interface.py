"""
Example: Using Tensor Interface
Demonstrates how to partition a design using tensor inputs
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partitioner3d import TensorInterface


def example_tensor_partition():
    """Example of partitioning from tensor inputs."""
    
    print("=" * 70)
    print("Example: 3D Partitioning from Tensor Inputs")
    print("=" * 70)
    
    # Initialize Tensor interface
    # Configure partition grid: 2x2x2 = 8 partitions
    interface = TensorInterface(
        num_partitions_x=2,
        num_partitions_y=2,
        num_partitions_z=2,
        use_terminal_awareness=True,
        optimization_mode="balanced",
        normalize_inputs=True
    )
    
    # Generate synthetic cell positions
    np.random.seed(42)
    num_cells = 20
    
    # Random positions in 3D space
    positions = np.random.rand(num_cells, 3) * 1000
    
    # Optional: cell sizes
    sizes = np.random.rand(num_cells, 3) * 50 + 10
    
    # Optional: terminal positions (I/O pins)
    terminal_positions = np.array([
        [0, 500, 0],      # Left edge
        [1000, 500, 0],   # Right edge
        [500, 0, 0],      # Bottom edge
        [500, 1000, 0]    # Top edge
    ])
    
    # Optional: cell names
    cell_names = [f"cell_{i}" for i in range(num_cells)]
    
    print(f"\nInput Data:")
    print(f"  Number of cells: {num_cells}")
    print(f"  Number of terminals: {len(terminal_positions)}")
    print(f"  Position range: X=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], "
          f"Y=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}], "
          f"Z=[{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}]")
    
    # Perform partitioning
    print("\nPerforming partitioning...")
    assignments, results = interface.partition_from_tensors(
        positions=positions,
        sizes=sizes,
        terminal_positions=terminal_positions,
        cell_names=cell_names
    )
    
    # Print summary
    print("\n" + interface.get_partition_summary(results))
    
    # Get partition masks (useful for downstream processing)
    print("\nGenerating partition masks...")
    masks = interface.get_partition_masks(assignments)
    
    print(f"Generated {len(masks)} binary masks (one per partition)")
    
    # Get partition centers
    centers = interface.get_partition_centers()
    print(f"\nPartition Centers:")
    for i, center in enumerate(centers):
        ix = i // (interface.partitioner.num_partitions_y * interface.partitioner.num_partitions_z)
        temp = i % (interface.partitioner.num_partitions_y * interface.partitioner.num_partitions_z)
        iy = temp // interface.partitioner.num_partitions_z
        iz = temp % interface.partitioner.num_partitions_z
        print(f"  Partition ({ix}, {iy}, {iz}): center at [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    
    # Convert to one-hot encoding (useful for ML applications)
    one_hot = interface.to_one_hot(assignments)
    print(f"\nOne-hot encoding shape: {one_hot.shape}")
    print(f"  (num_cells x num_partitions)")
    
    # Print some detailed assignments
    print("\nSample Cell Assignments:")
    print("-" * 70)
    for i in range(min(5, num_cells)):
        partition_id = tuple(assignments[i])
        print(f"  {cell_names[i]}: position={positions[i]}, assigned to partition {partition_id}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def example_batch_partition():
    """Example of batch partitioning multiple designs."""
    
    print("\n" + "=" * 70)
    print("Example: Batch Partitioning")
    print("=" * 70)
    
    interface = TensorInterface(
        num_partitions_x=2,
        num_partitions_y=2,
        num_partitions_z=1,
        normalize_inputs=True
    )
    
    # Generate multiple designs
    np.random.seed(42)
    batch_positions = [
        np.random.rand(10, 3) * 1000,
        np.random.rand(15, 3) * 1000,
        np.random.rand(12, 3) * 1000
    ]
    
    print(f"\nBatch processing {len(batch_positions)} designs:")
    for i, pos in enumerate(batch_positions):
        print(f"  Design {i+1}: {len(pos)} cells")
    
    # Partition all designs
    print("\nPerforming batch partitioning...")
    batch_results = interface.partition_batch(batch_positions)
    
    print(f"\nCompleted partitioning for {len(batch_results)} designs")
    
    for i, (assignments, results) in enumerate(batch_results):
        print(f"\nDesign {i+1} Results:")
        print(f"  Total cells: {results['num_cells']}")
        print(f"  Total partitions: {results['partition_statistics']['total_partitions']}")
        
        # Show distribution
        stats = results['partition_statistics']['cells_per_partition']
        partition_counts = [stats[pid] for pid in sorted(stats.keys())]
        print(f"  Partition distribution: {partition_counts}")
    
    print("\n" + "=" * 70)
    print("Batch example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    example_tensor_partition()
    example_batch_partition()

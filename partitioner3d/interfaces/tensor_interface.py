"""
Tensor Interface for 3D Partitioner
Provides interface to partition designs using tensor inputs
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from ..core.partitioner import Partitioner3D


class TensorInterface:
    """
    Interface for partitioning designs using tensor inputs.
    
    This interface handles:
    - Direct tensor input (numpy arrays or compatible formats)
    - Coordinate normalization
    - Partitioning invocation
    - Result formatting
    
    Suitable for integration with deep learning frameworks.
    """
    
    def __init__(
        self,
        num_partitions_x: int = 2,
        num_partitions_y: int = 2,
        num_partitions_z: int = 2,
        use_terminal_awareness: bool = True,
        optimization_mode: str = "balanced",
        normalize_inputs: bool = True
    ):
        """
        Initialize Tensor interface.
        
        Args:
            num_partitions_x: Number of partitions in X dimension
            num_partitions_y: Number of partitions in Y dimension
            num_partitions_z: Number of partitions in Z dimension (layers)
            use_terminal_awareness: Enable terminal-aware partitioning
            optimization_mode: Optimization mode ('balanced', 'wirelength', 'area')
            normalize_inputs: Automatically normalize input coordinates to [0, 1]
        """
        self.partitioner = Partitioner3D(
            num_partitions_x=num_partitions_x,
            num_partitions_y=num_partitions_y,
            num_partitions_z=num_partitions_z,
            use_terminal_awareness=use_terminal_awareness,
            optimization_mode=optimization_mode
        )
        
        self.normalize_inputs = normalize_inputs
        self.normalization_params = None
        
    def _normalize_coordinates(
        self,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize coordinates to [0, 1] range.
        
        Args:
            positions: Nx3 array of positions
            
        Returns:
            Tuple of (normalized_positions, normalization_params)
        """
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        
        # Avoid division by zero
        range_pos = max_pos - min_pos
        range_pos[range_pos == 0] = 1.0
        
        normalized = (positions - min_pos) / range_pos
        
        params = {
            'min': min_pos,
            'max': max_pos,
            'range': range_pos
        }
        
        return normalized, params
        
    def _denormalize_coordinates(
        self,
        positions: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """
        Denormalize coordinates back to original scale.
        
        Args:
            positions: Nx3 array of normalized positions
            params: Normalization parameters
            
        Returns:
            Denormalized positions
        """
        return positions * params['range'] + params['min']
        
    def partition_from_tensors(
        self,
        positions: Union[np.ndarray, list],
        sizes: Optional[Union[np.ndarray, list]] = None,
        terminal_positions: Optional[Union[np.ndarray, list]] = None,
        cell_names: Optional[list] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Partition using tensor inputs.
        
        Args:
            positions: Nx3 array or list of (x, y, z) positions
            sizes: Nx3 array or list of (width, height, depth) sizes
            terminal_positions: Mx3 array or list of terminal positions
            cell_names: Optional list of cell names for reference
            
        Returns:
            Tuple of (partition_assignments, results_dict)
        """
        # Convert to numpy arrays
        positions = np.array(positions, dtype=np.float32)
        
        if sizes is not None:
            sizes = np.array(sizes, dtype=np.float32)
        else:
            sizes = np.ones_like(positions)
            
        if terminal_positions is not None:
            terminal_positions = np.array(terminal_positions, dtype=np.float32)
            
        # Normalize if requested
        if self.normalize_inputs:
            positions, self.normalization_params = self._normalize_coordinates(positions)
            
            if sizes is not None:
                sizes = sizes / self.normalization_params['range']
                
            if terminal_positions is not None:
                terminal_positions = (
                    terminal_positions - self.normalization_params['min']
                ) / self.normalization_params['range']
                
        # Perform partitioning
        assignments, stats = self.partitioner.partition(
            positions=positions,
            sizes=sizes,
            terminal_positions=terminal_positions
        )
        
        # Format results
        results = {
            'num_cells': len(positions),
            'num_terminals': len(terminal_positions) if terminal_positions is not None else 0,
            'partition_assignments': assignments,
            'partition_statistics': stats,
            'partition_config': {
                'num_partitions_x': self.partitioner.num_partitions_x,
                'num_partitions_y': self.partitioner.num_partitions_y,
                'num_partitions_z': self.partitioner.num_partitions_z
            },
            'normalization_params': self.normalization_params
        }
        
        if cell_names is not None:
            results['cell_names'] = cell_names
            
        return assignments, results
        
    def partition_batch(
        self,
        batch_positions: list,
        batch_sizes: Optional[list] = None,
        batch_terminal_positions: Optional[list] = None
    ) -> list:
        """
        Partition multiple designs in batch.
        
        Args:
            batch_positions: List of Nx3 position arrays
            batch_sizes: List of Nx3 size arrays (optional)
            batch_terminal_positions: List of Mx3 terminal position arrays (optional)
            
        Returns:
            List of (assignments, results) tuples
        """
        results = []
        
        for i, positions in enumerate(batch_positions):
            sizes = batch_sizes[i] if batch_sizes is not None else None
            terminals = (
                batch_terminal_positions[i]
                if batch_terminal_positions is not None
                else None
            )
            
            assignments, result = self.partition_from_tensors(
                positions=positions,
                sizes=sizes,
                terminal_positions=terminals
            )
            
            results.append((assignments, result))
            
        return results
        
    def get_partition_masks(self, assignments: np.ndarray) -> Dict:
        """
        Get binary masks for each partition.
        
        Args:
            assignments: Nx3 array of partition assignments
            
        Returns:
            Dictionary mapping partition IDs to binary masks
        """
        n_cells = len(assignments)
        masks = {}
        
        for ix in range(self.partitioner.num_partitions_x):
            for iy in range(self.partitioner.num_partitions_y):
                for iz in range(self.partitioner.num_partitions_z):
                    partition_id = (ix, iy, iz)
                    
                    mask = (
                        (assignments[:, 0] == ix) &
                        (assignments[:, 1] == iy) &
                        (assignments[:, 2] == iz)
                    )
                    
                    masks[partition_id] = mask.astype(np.float32)
                    
        return masks
        
    def get_partition_centers(self) -> np.ndarray:
        """
        Get center coordinates of each partition.
        
        Returns:
            Px3 array of partition center positions
        """
        if self.partitioner.partition_boundaries is None:
            raise ValueError("Partitioning not performed yet.")
            
        centers = []
        
        for ix in range(self.partitioner.num_partitions_x):
            for iy in range(self.partitioner.num_partitions_y):
                for iz in range(self.partitioner.num_partitions_z):
                    x_bounds = self.partitioner.partition_boundaries['x']
                    y_bounds = self.partitioner.partition_boundaries['y']
                    z_bounds = self.partitioner.partition_boundaries['z']
                    
                    center_x = (x_bounds[ix] + x_bounds[ix + 1]) / 2
                    center_y = (y_bounds[iy] + y_bounds[iy + 1]) / 2
                    center_z = (z_bounds[iz] + z_bounds[iz + 1]) / 2
                    
                    centers.append([center_x, center_y, center_z])
                    
        return np.array(centers)
        
    def to_one_hot(self, assignments: np.ndarray) -> np.ndarray:
        """
        Convert partition assignments to one-hot encoding.
        
        Args:
            assignments: Nx3 array of partition indices
            
        Returns:
            NxP array where P is total number of partitions
        """
        n_cells = len(assignments)
        n_partitions = (
            self.partitioner.num_partitions_x *
            self.partitioner.num_partitions_y *
            self.partitioner.num_partitions_z
        )
        
        one_hot = np.zeros((n_cells, n_partitions), dtype=np.float32)
        
        for i, (ix, iy, iz) in enumerate(assignments):
            # Compute linear partition index
            partition_idx = (
                ix * self.partitioner.num_partitions_y * self.partitioner.num_partitions_z +
                iy * self.partitioner.num_partitions_z +
                iz
            )
            one_hot[i, partition_idx] = 1.0
            
        return one_hot
        
    def get_partition_summary(self, results: Dict) -> str:
        """
        Get a human-readable summary of the partitioning results.
        
        Args:
            results: Results dictionary from partition_from_tensors
            
        Returns:
            Formatted summary string
        """
        stats = results['partition_statistics']
        
        summary = []
        summary.append("=" * 60)
        summary.append("3D Tensor Partitioning Summary")
        summary.append("=" * 60)
        summary.append(f"Total Cells: {results['num_cells']}")
        summary.append(f"Total Terminals: {results['num_terminals']}")
        summary.append(f"\nPartition Configuration:")
        summary.append(f"  X partitions: {results['partition_config']['num_partitions_x']}")
        summary.append(f"  Y partitions: {results['partition_config']['num_partitions_y']}")
        summary.append(f"  Z partitions: {results['partition_config']['num_partitions_z']}")
        summary.append(f"  Total partitions: {stats['total_partitions']}")
        summary.append(f"\nCells per Partition:")
        
        for partition_id, count in stats['cells_per_partition'].items():
            summary.append(f"  Partition {partition_id}: {count} cells")
            
        if results.get('normalization_params'):
            summary.append(f"\nInput Normalization: Enabled")
            
        summary.append("=" * 60)
        
        return "\n".join(summary)

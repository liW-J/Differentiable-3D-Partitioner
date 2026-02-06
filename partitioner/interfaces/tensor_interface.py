"""
Tensor Interface for 3D Partitioner
Provides interface to partition designs using tensor inputs
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from ..core.partitioner import Partitioner


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

    def __init__(self,
                 use_terminal_awareness: bool = True,
                 optimization_mode: str = "balanced",
                 normalize_inputs: bool = True):
        """
        Initialize Tensor interface.
        
        Args:
            use_terminal_awareness: Enable terminal-aware partitioning
            optimization_mode: Optimization mode ('balanced', 'wirelength', 'area')
            normalize_inputs: Automatically normalize input coordinates to [0, 1]
        """
        self.partitioner = Partitioner(
            use_terminal_awareness=use_terminal_awareness,
            optimization_mode=optimization_mode)

        self.normalize_inputs = normalize_inputs
        self.normalization_params = None

    def _normalize_coordinates(
            self, positions: np.ndarray) -> Tuple[np.ndarray, Dict]:
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

        params = {'min': min_pos, 'max': max_pos, 'range': range_pos}

        return normalized, params

    def _denormalize_coordinates(self, positions: np.ndarray,
                                 params: Dict) -> np.ndarray:
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
            cell_names: Optional[list] = None) -> Tuple[np.ndarray, Dict]:
        """
        Partition using tensor inputs.
        
        Args:
            positions: Nx3 array or list of (x, y, z) positions
            sizes: Nx3 array or list of (width, height, depth) sizes.
                  If None, defaults to unit sizes (1, 1, 1) for all cells.
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
            positions, self.normalization_params = self._normalize_coordinates(
                positions)

            if sizes is not None:
                sizes = sizes / self.normalization_params['range']

            if terminal_positions is not None:
                terminal_positions = (terminal_positions -
                                      self.normalization_params['min']
                                      ) / self.normalization_params['range']

        # Perform partitioning
        assignments, stats = self.partitioner.partition(
            positions=positions,
            sizes=sizes,
            terminal_positions=terminal_positions)

        # Format results
        results = {
            'num_cells':
            len(positions),
            'num_terminals':
            len(terminal_positions) if terminal_positions is not None else 0,
            'partition_assignments':
            assignments,
            'partition_statistics':
            stats,
            'normalization_params':
            self.normalization_params
        }

        if cell_names is not None:
            results['cell_names'] = cell_names

        return assignments, results

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
        summary.append(f"  Total partitions: {stats['total_partitions']}")
        summary.append(f"\nCells per Partition:")

        for partition_id, count in stats['cells_per_partition'].items():
            summary.append(f"  Partition {partition_id}: {count} cells")

        if results.get('normalization_params'):
            summary.append(f"\nInput Normalization: Enabled")

        summary.append("=" * 60)

        return "\n".join(summary)

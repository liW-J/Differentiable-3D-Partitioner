"""
DEF Interface for 3D Partitioner
Provides interface to partition designs from DEF files
"""

import numpy as np
from typing import Dict, Optional, Tuple
from ..core.partitioner import Partitioner3D
from ..parsers.def_parser import DEFParser


class DEFInterface:
    """
    Interface for partitioning designs from DEF files.
    
    This interface handles:
    - DEF file parsing
    - Coordinate conversion
    - Partitioning invocation
    - Result formatting
    """
    
    def __init__(
        self,
        num_partitions_x: int = 2,
        num_partitions_y: int = 2,
        num_partitions_z: int = 2,
        use_terminal_awareness: bool = True,
        optimization_mode: str = "balanced"
    ):
        """
        Initialize DEF interface.
        
        Args:
            num_partitions_x: Number of partitions in X dimension
            num_partitions_y: Number of partitions in Y dimension
            num_partitions_z: Number of partitions in Z dimension (layers)
            use_terminal_awareness: Enable terminal-aware partitioning
            optimization_mode: Optimization mode ('balanced', 'wirelength', 'area')
        """
        self.partitioner = Partitioner3D(
            num_partitions_x=num_partitions_x,
            num_partitions_y=num_partitions_y,
            num_partitions_z=num_partitions_z,
            use_terminal_awareness=use_terminal_awareness,
            optimization_mode=optimization_mode
        )
        
        self.parser = DEFParser()
        self.parsed_data = None
        
    def load_def(self, filepath: str) -> Dict:
        """
        Load and parse a DEF file.
        
        Args:
            filepath: Path to the DEF file
            
        Returns:
            Parsed design data
        """
        self.parsed_data = self.parser.parse_file(filepath)
        return self.parsed_data
        
    def partition_from_file(
        self,
        filepath: str,
        use_pins_as_terminals: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Partition a design from a DEF file.
        
        Args:
            filepath: Path to the DEF file
            use_pins_as_terminals: Use pins as terminal positions
            
        Returns:
            Tuple of (partition_assignments, results_dict)
        """
        # Load and parse DEF file
        self.load_def(filepath)
        
        # Extract positions and sizes
        positions = self.parser.get_component_positions()
        sizes = self.parser.estimate_component_sizes()
        
        terminal_positions = None
        if use_pins_as_terminals:
            terminal_positions = self.parser.get_pin_positions()
            
        # Perform partitioning
        assignments, stats = self.partitioner.partition(
            positions=positions,
            sizes=sizes,
            terminal_positions=terminal_positions
        )
        
        # Format results
        results = {
            'design_name': self.parsed_data['design_name'],
            'num_components': len(self.parsed_data['components']),
            'num_pins': len(self.parsed_data['pins']),
            'partition_assignments': assignments,
            'partition_statistics': stats,
            'partition_config': {
                'num_partitions_x': self.partitioner.num_partitions_x,
                'num_partitions_y': self.partitioner.num_partitions_y,
                'num_partitions_z': self.partitioner.num_partitions_z
            }
        }
        
        return assignments, results
        
    def partition_from_content(
        self,
        def_content: str,
        use_pins_as_terminals: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Partition a design from DEF content string.
        
        Args:
            def_content: DEF file content as string
            use_pins_as_terminals: Use pins as terminal positions
            
        Returns:
            Tuple of (partition_assignments, results_dict)
        """
        # Parse DEF content
        self.parsed_data = self.parser.parse_content(def_content)
        
        # Extract positions and sizes
        positions = self.parser.get_component_positions()
        sizes = self.parser.estimate_component_sizes()
        
        terminal_positions = None
        if use_pins_as_terminals:
            terminal_positions = self.parser.get_pin_positions()
            
        # Perform partitioning
        assignments, stats = self.partitioner.partition(
            positions=positions,
            sizes=sizes,
            terminal_positions=terminal_positions
        )
        
        # Format results
        results = {
            'design_name': self.parsed_data['design_name'],
            'num_components': len(self.parsed_data['components']),
            'num_pins': len(self.parsed_data['pins']),
            'partition_assignments': assignments,
            'partition_statistics': stats,
            'partition_config': {
                'num_partitions_x': self.partitioner.num_partitions_x,
                'num_partitions_y': self.partitioner.num_partitions_y,
                'num_partitions_z': self.partitioner.num_partitions_z
            }
        }
        
        return assignments, results
        
    def export_partitioned_components(self, output_format: str = "dict") -> Dict:
        """
        Export partitioned components organized by partition.
        
        Args:
            output_format: Output format ('dict', 'list')
            
        Returns:
            Dictionary mapping partition IDs to component lists
        """
        if self.partitioner.partition_assignments is None:
            raise ValueError("No partitioning has been performed yet.")
            
        partitioned_components = {}
        
        for ix in range(self.partitioner.num_partitions_x):
            for iy in range(self.partitioner.num_partitions_y):
                for iz in range(self.partitioner.num_partitions_z):
                    partition_id = (ix, iy, iz)
                    cell_indices = self.partitioner.get_partition_cells(partition_id)
                    
                    components = []
                    for idx in cell_indices:
                        if idx < len(self.parsed_data['components']):
                            comp = self.parsed_data['components'][idx]
                            components.append({
                                'name': comp['name'],
                                'type': comp['cell_type'],
                                'position': comp['position'],
                                'partition': partition_id
                            })
                            
                    partitioned_components[partition_id] = components
                    
        return partitioned_components
        
    def get_partition_summary(self) -> str:
        """
        Get a human-readable summary of the partitioning results.
        
        Returns:
            Formatted summary string
        """
        if self.partitioner.partition_assignments is None:
            return "No partitioning performed yet."
            
        stats = self.partitioner.compute_partition_statistics()
        
        summary = []
        summary.append("=" * 60)
        summary.append("3D Partitioning Summary")
        summary.append("=" * 60)
        summary.append(f"Design: {self.parsed_data['design_name']}")
        summary.append(f"Total Components: {len(self.parsed_data['components'])}")
        summary.append(f"Total Pins: {len(self.parsed_data['pins'])}")
        summary.append(f"\nPartition Configuration:")
        summary.append(f"  X partitions: {self.partitioner.num_partitions_x}")
        summary.append(f"  Y partitions: {self.partitioner.num_partitions_y}")
        summary.append(f"  Z partitions: {self.partitioner.num_partitions_z}")
        summary.append(f"  Total partitions: {stats['total_partitions']}")
        summary.append(f"\nComponents per Partition:")
        
        for partition_id, count in stats['cells_per_partition'].items():
            summary.append(f"  Partition {partition_id}: {count} components")
            
        summary.append("=" * 60)
        
        return "\n".join(summary)

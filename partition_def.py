#!/usr/bin/env python3
"""
Command-line interface for 3D Partitioner
"""

import argparse
import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partitioner3d import DEFInterface


def main():
    parser = argparse.ArgumentParser(
        description='Partition a VLSI design from a DEF file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Partition with default 2x2x2 grid
  python partition_def.py design.def
  
  # Partition with custom grid
  python partition_def.py design.def -x 4 -y 4 -z 2
  
  # Disable terminal awareness
  python partition_def.py design.def --no-terminals
        """
    )
    
    parser.add_argument(
        'def_file',
        help='Path to the DEF file'
    )
    
    parser.add_argument(
        '-x', '--partitions-x',
        type=int,
        default=2,
        help='Number of partitions in X dimension (default: 2)'
    )
    
    parser.add_argument(
        '-y', '--partitions-y',
        type=int,
        default=2,
        help='Number of partitions in Y dimension (default: 2)'
    )
    
    parser.add_argument(
        '-z', '--partitions-z',
        type=int,
        default=2,
        help='Number of partitions in Z dimension (default: 2)'
    )
    
    parser.add_argument(
        '--no-terminals',
        action='store_true',
        help='Disable terminal-aware partitioning'
    )
    
    parser.add_argument(
        '-m', '--mode',
        choices=['balanced', 'wirelength', 'area'],
        default='balanced',
        help='Optimization mode (default: balanced)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file for partition assignments (optional)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.def_file):
        print(f"Error: DEF file '{args.def_file}' not found")
        return 1
    
    # Create interface
    print(f"Initializing 3D partitioner...")
    print(f"  Partition grid: {args.partitions_x}x{args.partitions_y}x{args.partitions_z}")
    print(f"  Terminal awareness: {not args.no_terminals}")
    print(f"  Optimization mode: {args.mode}")
    
    interface = DEFInterface(
        num_partitions_x=args.partitions_x,
        num_partitions_y=args.partitions_y,
        num_partitions_z=args.partitions_z,
        use_terminal_awareness=not args.no_terminals,
        optimization_mode=args.mode
    )
    
    # Partition the design
    print(f"\nLoading and partitioning: {args.def_file}")
    try:
        assignments, results = interface.partition_from_file(
            args.def_file,
            use_pins_as_terminals=not args.no_terminals
        )
    except Exception as e:
        print(f"Error during partitioning: {e}")
        return 1
    
    # Print summary
    print("\n" + interface.get_partition_summary())
    
    # Save output if requested
    if args.output:
        import json
        
        # Convert numpy types to native Python types
        output_data = {
            'design': results['design_name'],
            'partition_config': results['partition_config'],
            'assignments': assignments.tolist(),
            'statistics': {}
        }
        
        # Convert statistics (handle tuple keys)
        stats = results['partition_statistics']
        output_data['statistics']['total_partitions'] = stats['total_partitions']
        output_data['statistics']['cells_per_partition'] = {
            f"partition_{k[0]}_{k[1]}_{k[2]}": v 
            for k, v in stats['cells_per_partition'].items()
        }
        
        if 'partition_areas' in stats:
            output_data['statistics']['partition_areas'] = {
                f"partition_{k[0]}_{k[1]}_{k[2]}": v 
                for k, v in stats['partition_areas'].items()
            }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nPartition assignments saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

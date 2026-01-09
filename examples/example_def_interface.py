"""
Example: Using DEF Interface
Demonstrates how to partition a design from a DEF file
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partitioner3d import DEFInterface


def example_def_partition():
    """Example of partitioning from a DEF file."""
    
    print("=" * 70)
    print("Example: 3D Partitioning from DEF File")
    print("=" * 70)
    
    # Initialize DEF interface
    # Configure partition grid: 2x2x2 = 8 partitions
    interface = DEFInterface(
        num_partitions_x=2,
        num_partitions_y=2,
        num_partitions_z=2,
        use_terminal_awareness=True,
        optimization_mode="balanced"
    )
    
    # Path to DEF file
    def_file = os.path.join(os.path.dirname(__file__), "sample_design.def")
    
    if not os.path.exists(def_file):
        print(f"\nWarning: Sample DEF file not found at {def_file}")
        print("Please create a DEF file or use an existing one.")
        print("\nExample DEF content structure:")
        print("""
DESIGN sample_design ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 10000 ) ;

COMPONENTS 4 ;
- comp1 NAND2 + PLACED ( 1000 1000 ) N ;
- comp2 NOR2 + PLACED ( 3000 1000 ) N ;
- comp3 INV + PLACED ( 1000 3000 ) N ;
- comp4 BUF + PLACED ( 3000 3000 ) N ;
END COMPONENTS

PINS 2 ;
- in1 + NET in1 + DIRECTION INPUT + FIXED ( 0 5000 ) N ;
- out1 + NET out1 + DIRECTION OUTPUT + FIXED ( 10000 5000 ) N ;
END PINS

END DESIGN
        """)
        return
    
    print(f"\nLoading DEF file: {def_file}")
    
    # Partition the design
    assignments, results = interface.partition_from_file(
        filepath=def_file,
        use_pins_as_terminals=True
    )
    
    # Print summary
    print("\n" + interface.get_partition_summary())
    
    # Print detailed partition assignments
    print("\nDetailed Partition Assignments:")
    print("-" * 70)
    
    partitioned_comps = interface.export_partitioned_components()
    
    for partition_id, components in partitioned_comps.items():
        if components:
            print(f"\nPartition {partition_id}:")
            for comp in components:
                print(f"  - {comp['name']} ({comp['type']}) at {comp['position']}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    example_def_partition()

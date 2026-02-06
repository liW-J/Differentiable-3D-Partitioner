"""
Example: Using DEF Interface
Demonstrates how to partition a design from a DEF file
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partitioner import DEFInterface


def example_def_partition():
    """Example of partitioning from a DEF file."""

    print("=" * 70)
    print("Example: 3D Partitioning from DEF File")
    print("=" * 70)

    # Initialize DEF interface
    interface = DEFInterface(use_terminal_awareness=True,
                             optimization_mode="balanced")

    # Path to DEF file (project root / benchmark/def/sample_design.def)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def_file = os.path.join(project_root, "benchmark", "def",
                            "sample_design.def")

    if not os.path.exists(def_file):
        print(f"\nWarning: Sample DEF file not found at {def_file}")
        print("Please create a DEF file or use an existing one.")
        return

    print(f"\nLoading DEF file: {def_file}")

    # Partition the design
    assignments, results = interface.partition_from_file(
        filepath=def_file, use_pins_as_terminals=True)

    # Print summary
    print("\n" + interface.get_partition_summary())


    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    example_def_partition()

'''
Utilities for computing true cutsize from a DEF netlist and a partition assignment.

Supports two partition formats:
  - hMETIS: "cell_name partition_id" per line (e.g. results/partition-2d/jpeg/part.ub*.txt)
  - Ours  : one 0/1 per line (binary_assignment.txt) + companion node_names.txt
'''

import re
from pathlib import Path
from typing import Optional


def parse_def_netlist(def_path: str) -> dict:
    '''
    Parse the NETS section of a DEF file.

    Returns
    -------
    dict  {net_name: set_of_cell_names}
        Only cell pins are included; IO pins (PIN keyword) are skipped.
        Nets that connect to zero cells are excluded.
    '''
    netlist = {}
    in_nets = False
    current_net = None
    current_cells = set()

    net_start_re = re.compile(r'^-\s+(\S+)')
    pin_re = re.compile(r'\(\s+(\S+)\s+\S+\s+\)')

    with open(def_path, 'r', errors='replace') as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('NETS ') and stripped.endswith(';'):
                in_nets = True
                continue
            if stripped == 'END NETS':
                if current_net is not None and current_cells:
                    netlist[current_net] = current_cells
                break
            if not in_nets:
                continue

            # New net definition line
            m = net_start_re.match(stripped)
            if m:
                if current_net is not None and current_cells:
                    netlist[current_net] = current_cells
                current_net = m.group(1)
                current_cells = set()
                # Parse any inline pins on the same line
                rest = stripped[m.end():]
                for pm in pin_re.finditer(rest):
                    cell = pm.group(1)
                    if cell != 'PIN':
                        current_cells.add(cell)
                continue

            # End-of-net marker
            if stripped == ';':
                if current_net is not None and current_cells:
                    netlist[current_net] = current_cells
                current_net = None
                current_cells = set()
                continue

            # Continuation line with pin references
            if current_net is not None:
                for pm in pin_re.finditer(stripped):
                    cell = pm.group(1)
                    if cell != 'PIN':
                        current_cells.add(cell)

    return netlist


def compute_cutsize(netlist: dict, partition: dict) -> int:
    '''
    Count cut nets given a netlist and a partition assignment.

    Parameters
    ----------
    netlist   : {net_name: set_of_cell_names}  (from parse_def_netlist)
    partition : {cell_name: 0_or_1}

    A net is cut if its cells (that appear in partition) span both partitions.
    Cells not in partition are ignored for that net.
    Nets whose mapped cells are all in the same partition are not cut.
    '''
    cut = 0
    for cells in netlist.values():
        parts = set()
        for cell in cells:
            if cell in partition:
                parts.add(partition[cell])
            if len(parts) == 2:
                cut += 1
                break
    return cut


def load_hmetis_partition(part_txt_path: str) -> dict:
    '''
    Load an hMETIS partition file.

    Format: one line per cell
        cell_name  partition_id
    (partition_id is 0 or 1)
    '''
    partition = {}
    with open(part_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                cell_name = parts[0]
                pid = int(parts[1])
                partition[cell_name] = pid
    return partition


def load_ours_partition(binary_txt_path: str,
                        node_names_path: str) -> dict:
    '''
    Load our method's partition result.

    binary_txt_path : one 0/1 per line, in DREAMPlace node order
    node_names_path : one cell name per line, same order as binary_assignment
    '''
    with open(node_names_path, 'r') as f:
        node_names = [l.strip() for l in f if l.strip()]

    with open(binary_txt_path, 'r') as f:
        assignments = [int(l.strip()) for l in f if l.strip()]

    if len(node_names) != len(assignments):
        raise ValueError(
            f'Mismatch: {len(node_names)} node names vs '
            f'{len(assignments)} assignments in {binary_txt_path}')

    return {name: pid for name, pid in zip(node_names, assignments)}


def compute_cutsize_from_hgr(hgr_path: str, partition_indexed: list) -> int:
    '''
    Fast cutsize computation directly from a .hgr hypergraph file
    using an index-ordered partition list (list of 0/1, 1-indexed nodes).

    Useful when node ordering is known to match the hgr.
    '''
    cut = 0
    with open(hgr_path, 'r') as f:
        header = f.readline().split()
        fmt = int(header[2]) if len(header) >= 3 else 0
        has_node_weights = fmt in (10, 11)

        for line in f:
            line = line.strip()
            if not line:
                continue
            nodes = list(map(int, line.split()))
            parts = set(partition_indexed[n - 1] for n in nodes
                        if 1 <= n <= len(partition_indexed))
            if len(parts) == 2:
                cut += 1
    return cut


if __name__ == '__main__':
    import argparse, glob, os

    parser = argparse.ArgumentParser(
        description='Compute true cutsize from DEF netlist + partition files')
    parser.add_argument('--def', dest='def_path', required=True)
    parser.add_argument('--hmetis-dir', default=None,
                        help='Directory containing hMETIS part.*.txt files')
    parser.add_argument('--ours-dir', default=None,
                        help='Directory containing binary_assignment.txt + node_names.txt')
    args = parser.parse_args()

    print('Parsing DEF netlist ...')
    netlist = parse_def_netlist(args.def_path)
    print(f'  {len(netlist)} nets loaded')

    if args.hmetis_dir:
        part_files = sorted(glob.glob(os.path.join(args.hmetis_dir, 'part.*.txt')))
        print(f'\nhMETIS ({len(part_files)} files):')
        for pf in part_files:
            partition = load_hmetis_partition(pf)
            cs = compute_cutsize(netlist, partition)
            print(f'  {os.path.basename(pf)}: cutsize = {cs}')

    if args.ours_dir:
        binary = os.path.join(args.ours_dir, 'binary_assignment.txt')
        names  = os.path.join(args.ours_dir, 'node_names.txt')
        if os.path.exists(binary) and os.path.exists(names):
            partition = load_ours_partition(binary, names)
            cs = compute_cutsize(netlist, partition)
            print(f'\nOurs ({args.ours_dir}): cutsize = {cs}')
        else:
            print(f'\n[WARN] binary_assignment.txt or node_names.txt not found in {args.ours_dir}')

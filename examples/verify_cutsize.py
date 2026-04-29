"""Standalone script to compute true cutsize from DEF netlist + binary_assignment.txt.

Extracts node names via DREAMPlace PlaceDB directly (avoids importing the full
partitioner package which pulls in matplotlib).
"""
import sys
import os
import json
import re

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

_dreamplace_install = os.path.join(_project_root, 'thirdparty', 'DREAMPlace', 'install')
if os.path.exists(_dreamplace_install) and _dreamplace_install not in sys.path:
    sys.path.insert(0, _dreamplace_install)
_dreamplace_src = os.path.join(_dreamplace_install, 'dreamplace')
if _dreamplace_src not in sys.path:
    sys.path.insert(0, _dreamplace_src)

import thirdparty.DREAMPlace.dreamplace.PlaceDB as PlaceDB
import thirdparty.DREAMPlace.dreamplace.Params as Params


def parse_def_netlist(def_path):
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
            m = net_start_re.match(stripped)
            if m:
                if current_net is not None and current_cells:
                    netlist[current_net] = current_cells
                current_net = m.group(1)
                current_cells = set()
                rest = stripped[m.end():]
                for pm in pin_re.finditer(rest):
                    cell = pm.group(1)
                    if cell != 'PIN':
                        current_cells.add(cell)
                continue
            if stripped == ';':
                if current_net is not None and current_cells:
                    netlist[current_net] = current_cells
                current_net = None
                current_cells = set()
                continue
            if current_net is not None:
                for pm in pin_re.finditer(stripped):
                    cell = pm.group(1)
                    if cell != 'PIN':
                        current_cells.add(cell)
    return netlist


def compute_cutsize(netlist, partition):
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dreamplace', required=True)
    parser.add_argument('--def-file', required=True)
    parser.add_argument('--binary', required=True)
    args = parser.parse_args()

    print("1. Parsing design with DREAMPlace ...")
    params = Params.Params()
    params.load(args.dreamplace)
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    num_physical = placedb.num_physical_nodes
    node_names = [placedb.node_names[i].decode() for i in range(num_physical)]
    print(f"   num_physical_nodes = {num_physical}")
    print(f"   num_movable = {placedb.num_movable_nodes}, "
          f"num_terminals = {placedb.num_terminals}, "
          f"num_terminal_NIs = {placedb.num_terminal_NIs}")

    print("2. Loading binary_assignment.txt ...")
    with open(args.binary, 'r') as f:
        assignments = [int(l.strip()) for l in f if l.strip()]
    print(f"   assignments count = {len(assignments)}")

    if len(assignments) != len(node_names):
        print(f"   [WARN] mismatch: {len(node_names)} nodes vs {len(assignments)} assignments")
        min_len = min(len(node_names), len(assignments))
        node_names = node_names[:min_len]
        assignments = assignments[:min_len]

    partition = {name: pid for name, pid in zip(node_names, assignments)}
    top_count = sum(1 for v in partition.values() if v == 1)
    bot_count = sum(1 for v in partition.values() if v == 0)
    print(f"   Top cells: {top_count}, Bottom cells: {bot_count}")

    print("3. Parsing DEF netlist ...")
    netlist = parse_def_netlist(args.def_file)
    print(f"   {len(netlist)} nets loaded")

    print("4. Computing cutsize ...")
    cs = compute_cutsize(netlist, partition)
    print(f"\n   >>> TRUE CUTSIZE = {cs} <<<\n")


if __name__ == '__main__':
    main()

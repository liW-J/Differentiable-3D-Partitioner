'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2026-02-07 00:59:29
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-02-07 18:19:05
FilePath: /Differentiable-3D-Partitioner/examples/run_partitioner.py
Description: example for running the partitioner with file
'''

from partitioner import Differentiable3DPartitionerFlow, DreamplaceParser
import torch


def run_partitioner_with_file(dreamplace_config_file):
    parser = DreamplaceParser()
    parser.parse_design(dreamplace_config_file)

    flow = Differentiable3DPartitionerFlow(
        num_nodes=parser.num_nodes,
        num_nets=parser.num_nets,
        num_pins=parser.num_pins,
        node_pos=parser.node_pos,
        pin_pos=parser.pin_pos,
        flat_net2pin_map=parser.flat_net2pin_map,
        flat_net2pin_start_map=parser.flat_net2pin_start_map,
        pin2node_map=parser.pin2node_map,
        node_size_x=parser.node_size_x,
        node_size_y=parser.node_size_y,
    )

    flow.run()


def run_partitioner_with_tensor(node_pos, pin_pos, flat_net2pin_map,
                                flat_net2pin_start_map, pin2node_map,
                                node_size_x, node_size_y):
    flow = Differentiable3DPartitionerFlow(
        num_nodes=node_pos.shape[0],
        num_nets=flat_net2pin_map.shape[0],
        num_pins=pin_pos.shape[0],
        node_pos=node_pos,
        pin_pos=pin_pos,
        flat_net2pin_map=flat_net2pin_map,
        flat_net2pin_start_map=flat_net2pin_start_map,
        pin2node_map=pin2node_map,
        node_size_x=node_size_x,
        node_size_y=node_size_y,
    )
    flow.run()


if __name__ == "__main__":
    # run_partitioner_with_file(
    #     "benchmarks/bookself/iccad2022/case2_hidden/dreamplace.json")

    run_partitioner_with_file(
        "benchmarks/lefdef/nangate45/gcd/dreamplace.json")

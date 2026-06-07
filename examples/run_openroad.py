'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2026-02-07 00:59:29
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-02-12 03:49:35
FilePath: /Differentiable-3D-Partitioner/examples/run_partitioner.py
Description: example for running the partitioner with file
'''

from partitioner import Differentiable3DPartitionerFlow, DreamplaceParser
import torch
import multiprocessing as mp
import os


def run_partitioner_with_file(dreamplace_config_file, config_path = "configs/default.yaml"):
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
        dreamplace_basic=parser.dreamplace_basic,
        config_path=config_path,
        die_xl=parser.die_xl,
        die_yl=parser.die_yl,
        die_xh=parser.die_xh,
        die_yh=parser.die_yh
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


def run_task_on_gpu(dreamplace_config_file, config_path, gpu_id=None):
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)  # 当前进程绑定到指定 GPU

    run_partitioner_with_file(dreamplace_config_file, config_path)
    
if __name__ == "__main__":
    # 任务列表： (dreamplace_json, config_yaml, gpu_id)
    tasks = [
        # ("benchmarks/bookself/openroad/asap7/aes/dreamplace.json",
        # "configs/openroad/asap7/aes-bookself.yaml", 0),

        # ("benchmarks/bookself/openroad/asap7/ibex/dreamplace.json",
        # "configs/openroad/asap7/ibex-bookself.yaml", 1),

        ("benchmarks/bookself/openroad/asap7/jpeg/dreamplace.json",
        "configs/openroad/asap7/jpeg-bookself.yaml", 1),

        # ("benchmarks/bookself/openroad/asap7/swerv_wrapper/dreamplace.json",
        # "configs/openroad/asap7/swerv_wrapper-bookself.yaml", 1),

        # ("benchmarks/bookself/openroad/asap7/ariane133/dreamplace.json",
        # "configs/openroad/asap7/ariane133-bookself.yaml", 1),

        # ("benchmarks/bookself/openroad/asap7/bp_quad/dreamplace.json",
        # "configs/openroad/asap7/bp_quad-bookself.yaml", 1),
       
        
    ]

    def worker(args):
        dreamplace_config_file, config_path, gpu_id = args
        run_task_on_gpu(dreamplace_config_file, config_path, gpu_id)

    # 进程数可以设为 min(len(tasks), 物理 GPU 数)
    with mp.Pool(processes=len(tasks)) as pool:
        pool.map(worker, tasks)
    
    

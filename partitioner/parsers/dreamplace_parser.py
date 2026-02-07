'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2026-02-05 16:56:01
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-02-07 18:19:32
FilePath: /Differentiable-3D-Partitioner/partitioner/parsers/dreamplace_parser.py
Description: using DREAMPlace to parse the design
'''
import sys
import os

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_dreamplace_install_dir = os.path.join(_project_root, 'thirdparty',
                                       'DREAMPlace', 'install')
_dreamplace_source_dir = os.path.join(_dreamplace_install_dir, 'dreamplace')

if os.path.exists(_dreamplace_install_dir):
    if _dreamplace_install_dir not in sys.path:
        sys.path.insert(0, _dreamplace_install_dir)

if _dreamplace_source_dir not in sys.path:
    sys.path.insert(0, _dreamplace_source_dir)

import thirdparty.DREAMPlace.dreamplace.PlaceDB as PlaceDB
import thirdparty.DREAMPlace.dreamplace.Params as Params
import thirdparty.DREAMPlace.dreamplace.NonLinearPlace as NonLinearPlace


class DreamplaceParser:

    def __init__(self):
        self.num_nodes = 0
        self.num_nets = 0
        self.num_pins = 0

        self.node_pos = None
        self.pin_pos = None
        self.flat_net2pin_map = None
        self.flat_net2pin_start_map = None
        self.pin2node_map = None

        self.node_size_x = None
        self.node_size_y = None

    def parse_design(self, dreamplace_config_file):
        params = Params.Params()
        params.load(dreamplace_config_file)

        placedb = PlaceDB.PlaceDB()
        placedb(params)
        basic_place = NonLinearPlace.NonLinearPlace(params,
                                                    placedb,
                                                    timer=None)

        self.num_nodes = placedb.num_physical_nodes
        self.num_nets = placedb.num_nets
        self.num_pins = placedb.num_pins

        self.node_pos = basic_place.pos[0]
        self.pin_pos = basic_place.op_collections.pin_pos_op(self.node_pos)
        self.flat_net2pin_map = basic_place.data_collections.flat_net2pin_map
        self.flat_net2pin_start_map = basic_place.data_collections.flat_net2pin_start_map
        self.pin2node_map = basic_place.data_collections.pin2node_map
        self.node_size_x = basic_place.data_collections.node_size_x
        self.node_size_y = basic_place.data_collections.node_size_y


if __name__ == "__main__":
    parser = DreamplaceParser()
    parser.parse_design(
        "benchmarks/bookself/iccad2022/case2_hidden/dreamplace.json")

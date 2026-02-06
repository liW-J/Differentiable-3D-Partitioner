'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2026-02-05 16:56:01
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-02-07 00:40:16
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
        self.num_cells = 0
        self.num_nets = 0
        self.num_pins = 0

        self.node_pos = None
        self.pin_pos = None
        self.flat_net2pin_map = None
        self.flat_net2pin_start_map = None
        self.pin2node_map = None

        self.node_size_x = None
        self.node_size_y = None

    def parse_design(self, params: Params):

        placedb = PlaceDB()
        placedb(params)
        basic_place = NonLinearPlace.NonLinearPlace(params,
                                                    placedb,
                                                    timer=None)

        self.num_cells = placedb.num_nodes
        self.num_nets = placedb.num_nets
        self.num_pins = placedb.num_pins
        self.node_pos = placedb.node_x
        self.pin_pos = placedb.pin_x
        self.flat_net2pin_map = placedb.flat_net2pin_map
        self.flat_net2pin_start_map = placedb.flat_net2pin_start_map
        self.pin2node_map = placedb.pin2node_map
        self.node_size_x = placedb.node_size_x
        self.node_size_y = placedb.node_size_y


if __name__ == "__main__":
    params = Params.Params()
    parser = DreamplaceParser()

    params.load(sys.argv[1])
    parser.parse_design(params)


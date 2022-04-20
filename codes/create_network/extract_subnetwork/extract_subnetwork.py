#!/usr/bin/env python
# coding: utf-8
# %%
# Copyright (c) 2021 Takaki Yamamoto
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract subnetworks for attribution for NSP (bidirectional) and SP (unidirectional) models"""

from functions import system_utility as sutil
import numpy as np
import os
import sys
version = 4
print("version=%d" % version)



# %%
######### Load parameters from yml file #########

filename_yml = 'input_create_network.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

exp_network = yaml_obj["network_name"]
time_min = yaml_obj["time_min"]
time_max = yaml_obj["time_max"]
shuffle_switch = yaml_obj["shuffle_switch"]


#### load conditions #####

suffix = "-FLedit"  # added to the network name
network_num = exp_network

base_path_file = "base_path.txt"
base_path = sutil.Read1LineText(base_path_file)

total_frame = np.loadtxt(base_path + "total_frame.txt").astype(np.int64)


crop_size_file = "crop_size.txt"
crop_size_array = np.loadtxt(crop_size_file).astype(np.int16)
crop_width = crop_size_array[0]
crop_height = crop_size_array[1]
crop_shift_height = crop_size_array[2]


shuffle_list = [0, 1]
if shuffle_switch == 0:
    shuffle_list = [0]


# %%
#### Define dir and file names #####

analdir_name = "analysis/"
analdir_path = base_path + analdir_name

dir_network = analdir_path + \
    "network%s_crop_w=%d_h=%d/" % (network_num, crop_width, crop_height)


new_network_num = network_num
new_dir_network = analdir_path + \
    "network%s_crop_w=%d_h=%d/" % (new_network_num, crop_width, crop_height)


# %%
def OutputTargetCellID(g, nodeID, num_time, frame, ModelType):

    debug = 0  # We set debug=1, when debugging.

    time_list = []
    for i in range(num_time):
        time_list.append("t"+str(i))

    if debug == 1:
        print("\ntarget cellID = %d" % nodeID)

    AllNeighborCellList = [[] for _ in range(num_time)]   # t0,t1,t2,t3,,,

    AllTargetCellList = [[] for _ in range(num_time)]   # t0,t1,t2,t3,,,

    AllTargetCellListNoDoubling = [[]
                                   for _ in range(num_time)]   # t0,t1,t2,t3,,,

    ### MainLineage of the target cell ####

    LineageList = []
    LineageList.append(nodeID)
    #count_time = 0
    if num_time >= 2:  # when num_time = 1, we don't need to trace the lineage
        for count_time, time_name in enumerate(reversed(time_list[1:])):

            # in reversed direction
            time_name_target = time_list[num_time-count_time-1]
            time_name_next = time_list[num_time-count_time-2]
            etype = (time_name_target, 'time_rev', time_name_next)

            outdegree = g.out_degrees(nodeID, etype=etype)

            if outdegree == 1:
                pair = g.out_edges(nodeID, form='uv', etype=etype)

                nodeID = pair[1].tolist()
                if debug == 1:
                    print(nodeID)

                LineageList.extend(nodeID)
            else:
                break

    if debug == 1:
        print("MainLineage")
        print(LineageList)

    for i in range(len(LineageList)):
        AllTargetCellList[num_time-i-1].append(LineageList[i])

    if debug == 1:
        print("MainLineage is added")
        print(AllTargetCellList)

    ### Neighbor cells of MainLineage cells of the target cell ####

    #countLineage = 0
    for countLineage, nodeID_Lineage in enumerate(LineageList):  # go backward

        time_name_target = time_list[num_time-countLineage-1]

        etype = (time_name_target, 'interaction', time_name_target)

        pair = g.out_edges(nodeID_Lineage, form='uv', etype=etype)

        NeighborIDList = pair[1].to('cpu').detach().numpy().copy()

        AllTargetCellList[num_time-countLineage-1].extend(NeighborIDList)

        AllNeighborCellList[num_time-countLineage-1].extend(NeighborIDList)

        if debug == 1:
            print("Neghbor cells of %s are added" % time_name_target)
            print(AllTargetCellList)

        if (countLineage >= 1) & (ModelType == "NSP"):  # if not the final plane

            MainLineageCellID = LineageList[countLineage]

            #time_name = time_list[-1]
            time_name_target = time_list[num_time-countLineage-1]
            time_name_next = time_list[num_time-countLineage]
            etype = (time_name_target, 'time', time_name_next)

            if debug == 1:
                print(etype)

            if debug == 1:
                print("Next cell in the MainLineage")
            pair1 = g.out_edges(MainLineageCellID, form='uv', etype=etype)
            if debug == 1:
                print(pair1[1].tolist())

            AllTargetCellList[num_time-countLineage].extend(pair1[1].tolist())

            pair2 = g.out_edges(NeighborIDList, form='uv', etype=etype)
            if debug == 1:
                print("Next cell of neighbor cells")

            targetCellIDListTemporal = pair2[1].tolist()
            if debug == 1:
                print(targetCellIDListTemporal)
            AllTargetCellList[num_time -
                              countLineage].extend(targetCellIDListTemporal)

            if debug == 1:
                print(
                    "Next cell in the MainLineage and  Next cell of neighbor cells are added")
                print(AllTargetCellList)

            if countLineage >= 2:

                for j in range(countLineage-1):

                    time_name_target = time_list[num_time-countLineage-1+j+1]
                    time_name_next = time_list[num_time-countLineage+j+1]
                    etype = (time_name_target, 'time', time_name_next)
                    if debug == 1:
                        print("Succeeding temporal edge")
                        print(etype)

                    if len(targetCellIDListTemporal) >= 1:

                        pair_succeeding = g.out_edges(
                            targetCellIDListTemporal, form='uv', etype=etype)
                        if debug == 1:
                            print(pair_succeeding)
                        targetCellIDListTemporal = pair_succeeding[1]

                        targetCellIDListTemporal = targetCellIDListTemporal.tolist()
                        AllTargetCellList[num_time-countLineage +
                                          j+1].extend(targetCellIDListTemporal)
                        if debug == 1:
                            print("Succeeding lineage is added")
                            print(AllTargetCellList)

        ### Remove doubling #####

        for i in range(num_time):

            AllTargetCellListNoDoubling[i] = list(set(AllTargetCellList[i]))

    if debug == 1:
        print("All cells with doubling")
        print(AllTargetCellList)

        print("All cells without doubling")
        print(AllTargetCellListNoDoubling)

        print("All neighbor cell list")
        print(AllNeighborCellList)

    return AllTargetCellListNoDoubling, AllNeighborCellList, LineageList


# %%
#ModelType = "NSP"

for ModelType in ["NSP", "SP"]:
    for shuffle in shuffle_list:
        for num_time in range(time_min, time_max+1):

            new_networkdir_name = "network%s_num_w=%d_h=%d_time=%d/" % (
                new_network_num, crop_width, crop_height, num_time)
            new_networkdir_path = new_dir_network + new_networkdir_name

            new_cellIDdir_noborder_name = "network%s_cellID_FinalLayer_noborder_num_w=%d_h=%d_time=%d" % (
                new_network_num, crop_width, crop_height, num_time)
            new_cellIDdir_noborder_path = new_dir_network + new_cellIDdir_noborder_name

            ### Directory for target cell IDs #######

            AllTargetCellListNoDoubling_dirname = "network%s_%s_AllTargetCellListNoDoubling_noborder_num_w=%d_h=%d_time=%d/" % (
                network_num, ModelType, crop_width, crop_height, num_time)
            AllTargetCellListNoDoubling_dirpath = new_dir_network + \
                AllTargetCellListNoDoubling_dirname

            LineageList_dirname = "network%s_%s_LineageList_noborder_num_w=%d_h=%d_time=%d/" % (
                network_num, ModelType, crop_width, crop_height, num_time)
            LineageList_dirpath = new_dir_network + LineageList_dirname

            AllNeighborCellList_dirname = "network%s_%s_AllNeighborCellList_noborder_num_w=%d_h=%d_time=%d/" % (
                network_num, ModelType, crop_width, crop_height, num_time)
            AllNeighborCellList_dirpath = new_dir_network + AllNeighborCellList_dirname

            sutil.MakeDirs(AllTargetCellListNoDoubling_dirpath)
            sutil.MakeDirs(LineageList_dirpath)
            sutil.MakeDirs(AllNeighborCellList_dirpath)

            if num_time >= 1:  # Modified to deal with all time frames
                for frame in range(total_frame+1-num_time-1):

                    if shuffle == 0:
                        filename = "NetworkWithFeartures_t=%dto%d.pickle" % (
                            frame, frame+num_time-1)

                        g = sutil.PickleLoad(new_networkdir_path + filename)

                        filename = "CellID_FinalLayer_t=%dto%d.txt" % (
                            frame, frame+num_time-1)

                        IDListFinal = np.loadtxt(
                            new_cellIDdir_noborder_path + "/" + filename)

                        Frame_AllTargetCellListNoDoubling_dirpath = AllTargetCellListNoDoubling_dirpath + \
                            "t=%dto%d/" % (frame, frame+num_time-1)
                        sutil.MakeDirs(
                            Frame_AllTargetCellListNoDoubling_dirpath)

                        Frame_LineageList_dirpath = LineageList_dirpath + \
                            "t=%dto%d/" % (frame, frame+num_time-1)
                        sutil.MakeDirs(Frame_LineageList_dirpath)

                        Frame_AllNeighborCellList_dirpath = AllNeighborCellList_dirpath + \
                            "t=%dto%d/" % (frame, frame+num_time-1)
                        sutil.MakeDirs(Frame_AllNeighborCellList_dirpath)

                        # search for the cells at the final layer
                        for nodeID in IDListFinal:

                            # print(num_time)
                            nodeID = int(nodeID)
                            # print(nodeID)
                            AllTargetCellListNoDoubling, AllNeighborCellList, LineageList = OutputTargetCellID(
                                g, nodeID, num_time, frame, ModelType)

                            filename = "AllTargetCellListNoDoubling_cellID=%d_t=%dto%d.pickle" % (
                                nodeID, frame, frame+num_time-1)
                            savepath = Frame_AllTargetCellListNoDoubling_dirpath + filename
                            sutil.PickleDump(
                                AllTargetCellListNoDoubling, savepath)

                            filename = "LineageList_cellID=%d_t=%dto%d.pickle" % (
                                nodeID, frame, frame+num_time-1)
                            savepath = Frame_LineageList_dirpath + filename
                            sutil.PickleDump(LineageList, savepath)

                            filename = "AllNeighborCellList_cellID=%d_t=%dto%d.pickle" % (
                                nodeID, frame, frame+num_time-1)
                            savepath = Frame_AllNeighborCellList_dirpath + filename
                            sutil.PickleDump(AllNeighborCellList, savepath)

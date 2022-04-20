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

"""Load created graphs and add G1 per area """

import numpy as np
import torch as th
import os
import sys
from functions import system_utility as sutil
from distutils.dir_util import copy_tree
version = 2
print("version=%d" % version)



voronoi_thr = 0.001


# %%


def IntArrayToOneHotVector(target_vector, n_labels):

    n_labels = np.int(n_labels)
    # print(n_labels)
    # print(n_labels.dtype)
    # print(target_vector)

    target_vector = target_vector.astype(np.int64)
    onehot = np.eye(n_labels)[target_vector]
    return onehot[:, 0]


def Standardization(data, mean, sigma):

    std_data = (data-mean)/sigma
    return std_data


def Normarization(data, min_val, max_val):

    norm_data = (data-min_val)/(max_val-min_val)
    return norm_data


# %%
#### load conditions #####
#edge_feature_name_list = ["EdgeLength_pix","Rank1ForSrc", "Rank2ForSrc", "Rank3ForSrc","Rank1ForDst", "Rank2ForDst", "Rank3ForDst"]

suffix = "-G1Signal"  # added to the network name
name_column = "G1Signal"


######### Load parameters from yml file #########

filename_yml = 'input_load_graph_add_G1perArea.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

exp_network = yaml_obj["network_name"]
time_min = yaml_obj["time_min"]
time_max = yaml_obj["time_max"]
shuffle_switch = yaml_obj["shuffle_switch"]
# exp_sim_switch = yaml_obj["exp_sim_switch"]# if 1, experiment. if 0, simulation

#tif_size = yaml_obj["tif_size"]

network_num_track = yaml_obj["network_num_track"]


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


new_network_num = network_num + suffix
new_dir_network = analdir_path + \
    "network%s_crop_w=%d_h=%d/" % (new_network_num, crop_width, crop_height)
sutil.MakeDirs(new_dir_network)

new_dir_network_info = "network_info/"

new_dir_network_info_path = new_dir_network + new_dir_network_info
sutil.MakeDirs(new_dir_network_info_path)

# print(new_dir_network_info_path)


# %%
new_column_num = 10
new_total_column = new_column_num + 1

index_list_minmax = [4, 5, 6, 7, 8, 9]  # 9 is the final index

name_list_minmax = ["X",
                    "Y",
                    "Voronoi_area",
                    "G1_voronoi",
                    "Area",
                    "G1"]


if len(index_list_minmax) != len(name_list_minmax):

    print("Error: Number of list")


# %%
# Calculate minimum Integrated G1 marker intensity from cropped track data
#######Read npy files in "min_max_list.txt" #########
# Index of npy files starts from 1
# Trace cells in "trackdata_crop_w=%d_h=%d_%d.txt" in "trackdata_crop_w=%d_h=%d/"%(crop_width,crop_height).
# Extrating data from lineage npy file only for the traced cells, and then set variables zero if voronoi area =0.
# Then save the cropped lineage as "trackdata_crop_w=%d_h=%d_%d.txt"%( crop_width,  crop_height, frame) at
# network%s_crop_w=%d_h=%d/"%(network_num, crop_width,crop_height)/
# network%s_trackdata_crop_w=%d_h=%d/"%(network_num, crop_width,crop_height)

# All the cells in the cropped area in min_max_list.txt" is allocated to "data_all".
# The data is from trackdata_crop_w=%d_h=%d_%d.txt" in "trackdata_crop_w=%d_h=%d/"%(crop_width,crop_height) without the zero filling.
# data_all_new is the array for zero-filled data for voronoi area zero cells.

base_dir_list = sutil.ReadLinesText("base_path_list.txt")


data_all = np.empty((0, 10))

data_all_new = np.empty((0, new_total_column))

G1_min_list = []
for base_path in base_dir_list:

    analdir_name = "analysis/"
    analdir_path = base_path + analdir_name

    dir_network = analdir_path + \
        "network%s_crop_w=%d_h=%d/" % (network_num_track,
                                       crop_width, crop_height)
    trackdata_cropdir_name = "network%s_trackdata_crop_w=%d_h=%d/" % (
        network_num_track, crop_width, crop_height)
    trackdata_cropdir_path = dir_network + trackdata_cropdir_name

    for frame in range(total_frame):
        filename_load = "trackdata_crop_w=%d_h=%d_%d.txt" % (
            crop_width,  crop_height, frame)
        filepath_load = trackdata_cropdir_path + filename_load

        data_frame = np.loadtxt(filepath_load)

        # Ignore true_divide warning #Put very large value for Nan to calculate real minimum of G1 per area
        G1_min_tmp = np.min(np.nan_to_num(
            data_frame[:, 9]/data_frame[:, 8], nan=1000000))

        if G1_min_tmp == 0:
            print("Zero min frame")
            # print(base_path)
            # print(frame)
            #print(np.argmin(data_frame[:, 9]))

        G1_min_list.append(G1_min_tmp)


G1_min = np.min(np.array(G1_min_list))


# %%
#######Read npy files in "min_max_list.txt" #########
# Index of npy files starts from 1
# Trace cells in "trackdata_crop_w=%d_h=%d_%d.txt" in "trackdata_crop_w=%d_h=%d/"%(crop_width,crop_height).
# Extrating data from lineage npy file only for the traced cells, and then set variables zero if voronoi area =0.
# Then save the cropped lineage as "trackdata_crop_w=%d_h=%d_%d.txt"%( crop_width,  crop_height, frame) at
# network%s_crop_w=%d_h=%d/"%(network_num, crop_width,crop_height)/
# network%s_trackdata_crop_w=%d_h=%d/"%(network_num, crop_width,crop_height)

# All the cells in the cropped area in min_max_list.txt" is allocated to "data_all".
# The data is from trackdata_crop_w=%d_h=%d_%d.txt" in "trackdata_crop_w=%d_h=%d/"%(crop_width,crop_height) without the zero filling.
# data_all_new is the array for zero-filled data for voronoi area zero cells.

base_dir_list = sutil.ReadLinesText("base_path_list.txt")


data_all = np.empty((0, 10))

data_all_new = np.empty((0, new_total_column))


for base_path in base_dir_list:

    analdir_name = "analysis/"
    analdir_path = base_path + analdir_name

    dir_network = analdir_path + \
        "network%s_crop_w=%d_h=%d/" % (network_num_track,
                                       crop_width, crop_height)
    trackdata_cropdir_name = "network%s_trackdata_crop_w=%d_h=%d/" % (
        network_num_track, crop_width, crop_height)
    trackdata_cropdir_path = dir_network + trackdata_cropdir_name

    new_dir_network = analdir_path + \
        "network%s_crop_w=%d_h=%d/" % (new_network_num,
                                       crop_width, crop_height)

    trackdata_cropdir_name_new = "network%s_trackdata_crop_w=%d_h=%d/" % (
        new_network_num, crop_width, crop_height)
    trackdata_cropdir_path_new = new_dir_network + trackdata_cropdir_name_new

    sutil.MakeDirs(trackdata_cropdir_path_new)

    for frame in range(total_frame):
        filename_load = "trackdata_crop_w=%d_h=%d_%d.txt" % (
            crop_width,  crop_height, frame)
        filepath_load = trackdata_cropdir_path + filename_load

        filename_save = "trackdata_crop_w=%d_h=%d_%d.txt" % (
            crop_width,  crop_height, frame)
        filepath_save = trackdata_cropdir_path_new + filename_save

        data_frame = np.loadtxt(filepath_load)

        ##############################
        ### New column  G1/Area###########
        new_column_data = (np.nan_to_num(data_frame[:, 9]/data_frame[:, 8], nan=G1_min) - G1_min)*np.nan_to_num(
            data_frame[:, 8])  # Ignore true_divide warning #Put real minimum value for voronoi-zero cell

        new_column_data = np.reshape(
            new_column_data, (new_column_data.shape[0], 1))

        data_frame_new = np.append(data_frame, new_column_data, axis=1)

        np.savetxt(filepath_save, data_frame_new)

        data_all = np.append(data_all, data_frame, axis=0)

        data_all_new = np.append(data_all_new, data_frame_new, axis=0)


# %%
new_column = data_all_new[:, new_column_num]
target_max = np.max(new_column)
target_min = np.min(new_column)
# print(np.max(new_column))
# print(np.min(new_column))

np.savetxt(new_dir_network_info_path +
           "G1Signal_max.txt", np.array([target_max]))

np.savetxt(new_dir_network_info_path + "G1Signal_min.txt",
           np.array([target_min]))  # Remove voronoi zer ocell

# print(np.argmax(new_column))
# print(data_all_new[np.argmax(new_column)])


#print("Max and min")
# print(target_max)
# print(target_min)


# %%
base_path_file = "base_path.txt"
base_path = sutil.Read1LineText(base_path_file)

analdir_name = "analysis/"
analdir_path = base_path + analdir_name

new_network_num = network_num + suffix
new_dir_network = analdir_path + \
    "network%s_crop_w=%d_h=%d/" % (new_network_num, crop_width, crop_height)


dir_network = analdir_path + \
    "network%s_crop_w=%d_h=%d/" % (network_num, crop_width, crop_height)


# %%
# Define edge features


#margin = 20
# for shuffle in [0,1]:
for shuffle in shuffle_list:
    for num_time in range(time_min, time_max+1):
        # for num_time in [4]:
        time_list = []

        for i in range(num_time):
            time_list.append("t"+str(i))

        networkdir_name = "network%s_num_w=%d_h=%d_time=%d/" % (
            network_num, crop_width, crop_height, num_time)
        networkdir_path = dir_network + networkdir_name

        new_networkdir_name = "network%s_num_w=%d_h=%d_time=%d/" % (
            new_network_num, crop_width, crop_height, num_time)
        new_networkdir_path = new_dir_network + new_networkdir_name

        sutil.MakeDirs(new_networkdir_path)

        #### File copy  #####
        cellIDdir_noborder_name = "network%s_cellID_FinalLayer_noborder_num_w=%d_h=%d_time=%d" % (
            network_num, crop_width, crop_height, num_time)
        cellIDdir_noborder_path = dir_network + cellIDdir_noborder_name

        new_cellIDdir_noborder_name = "network%s_cellID_FinalLayer_noborder_num_w=%d_h=%d_time=%d" % (
            new_network_num, crop_width, crop_height, num_time)
        new_cellIDdir_noborder_path = new_dir_network + new_cellIDdir_noborder_name
        sutil.MakeDirs(new_cellIDdir_noborder_path)

        copy_tree(cellIDdir_noborder_path, new_cellIDdir_noborder_path)

        for frame in range(total_frame+1-num_time-1):

            if shuffle == 0:
                filename = "NetworkWithFeartures_t=%dto%d.pickle" % (
                    frame, frame+num_time-1)

                g = sutil.PickleLoad(networkdir_path + filename)

            if shuffle == 1:
                filename = "NetworkWithFeartures_t=%dto%d.pickle" % (
                    frame, frame+num_time-1)

                g = sutil.PickleLoad(shuffle_networkdir_path + filename)

            #####  Add  #######

            for i in range(num_time):

                #print("frame=%d, time=%d"%(frame,i))

                time_label = time_list[i]

                num_node = g.num_nodes(time_label)
                # print("num_node=%d"%num_node)

                trackdata_cropdir_name_new = "network%s_trackdata_crop_w=%d_h=%d/" % (
                    new_network_num, crop_width, crop_height)
                trackdata_cropdir_path_new = new_dir_network + trackdata_cropdir_name_new

                trackfile_name = "trackdata_crop_w=%d_h=%d_%d.txt" % (
                    crop_width, crop_height, frame + i)
                track_load = np.loadtxt(
                    trackdata_cropdir_path_new + trackfile_name)

                voronoi = track_load[:, 6]

                # for 1st frame of  each network, we need to eliminate the div-dif cell because we don't have the cells in the network.

                if i == 0:
                    # remove voronoi = 0 cells
                    track_load = track_load[np.where(voronoi > voronoi_thr)]

                #print(trackdata_cropdir_path + trackfile_name)

                # This include voronoi zero cell when i>0
                CellID = track_load[:, 3].astype(np.int64)

                column_n = new_column_num
                temp_data = track_load[:, column_n]
                temp_raw = np.zeros((num_node, 1))
                temp_raw[CellID, 0] = temp_data
                temp_norm_tmp = (temp_data-target_min)/(target_max-target_min)

                if np.min(temp_norm_tmp) < 0:
                    print("Normalization error:min")
                    print(np.min(temp_norm_tmp))
                if np.min(temp_norm_tmp) > 1.0:
                    print("Normalization error:max")
                    print(np.max(temp_norm_tmp))

                temp_norm = np.zeros((num_node, 1))
                # For voronoi zero cell, the value is set to zero
                temp_norm[CellID, 0] = temp_norm_tmp

                name_raw = name_column + "_raw"
                name_norm = name_column + "_norm"
                g.nodes[time_label].data[name_raw] = th.tensor(
                    temp_raw, dtype=th.float)
                g.nodes[time_label].data[name_norm] = th.tensor(
                    temp_norm, dtype=th.float)

            ####### Save the network #######

            filename = "NetworkWithFeartures_t=%dto%d.pickle" % (
                frame, frame+num_time-1)
            #save_graphs(networkdir_path + filename, g)

            sutil.PickleDump(g, new_networkdir_path + filename)


# %%

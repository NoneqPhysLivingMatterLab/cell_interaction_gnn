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

"""Detect neighbors from linieage npy file"""

import tifffile
from functions import system_utility as sutil
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import dgl
import os
import sys
version = 20
print("version=%d" % version)



# parameters for plot
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 0.75
plt.rcParams['ytick.major.width'] = 0.75
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.75
# plt.locator_params(axis='x',nbins=20)
# plt.locator_params(axis='y',nbins=6)


figx = 3.14
figy = 3.14
left = 0.2
right = 0.9
bottom = 0.2
top = 0.8
dpi = 300

voronoi_thr = 0.001


# %%
########## Choose features, and the index in the npy file ########
filename_yml = 'input_create_network.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

index_list_minmax = yaml_obj["index_list_minmax"]
name_list_minmax = yaml_obj["name_list_minmax"]

if len(index_list_minmax) != len(name_list_minmax):

    print("Error: Number of list")


# %%
# Define functions
def IntArrayToOneHotVector(target_vector, n_labels):

    n_labels = np.int(n_labels)

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
######### Load parameters from yml file #########
filename_yml = 'input_create_network.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

exp_network = yaml_obj["network_name"]
time_min = yaml_obj["time_min"]
time_max = yaml_obj["time_max"]
shuffle_switch = yaml_obj["shuffle_switch"]


######### Load parameters from text files #########


new_lineagefile = sutil.ReadLinesText("lineage_filename_all.txt")
# print(new_lineagefile)

network_num = exp_network

shuffle_list = [0, 1]
if shuffle_switch == 0:
    shuffle_list = [0]


base_path_file = "base_path.txt"
base_path = sutil.Read1LineText(base_path_file)


total_frame = np.loadtxt(base_path + "total_frame.txt").astype(np.int64)


crop_size_list = np.loadtxt(
    "crop_size_list.txt", delimiter=",").astype(np.int16)

# print(crop_size_list)


# %%
####### Read files in base_path_list.txt to calculate max values. #####
# Trace cells in "trackdata_crop_w=%d_h=%d_%d.txt" in "trackdata_crop_w=%d_h=%d/"%(crop_width,crop_height).
# Extrating data from lineage npy file only for the traced cells, and then set variables zero if voronoi area =0.
# Then save the cropped lineage as "trackdata_crop_w=%d_h=%d_%d.txt"%( crop_width,  crop_height, frame) at
# network%s_crop_w=%d_h=%d/"%(network_num, crop_width,crop_height)/
# network%s_trackdata_crop_w=%d_h=%d/"%(network_num, crop_width,crop_height)

# All the cells in the cropped area in base_path_list.txt" is allocated to "data_all".
# The data is from trackdata_crop_w=%d_h=%d_%d.txt" in "trackdata_crop_w=%d_h=%d/"%(crop_width,crop_height) without the zero filling.
# data_all_new is the array for zero-filled data for voronoi area zero cells.

base_dir_list = sutil.ReadLinesText("base_path_list.txt")


data_all = np.empty((0, 10))


count = 0
for base_path in base_dir_list:

    total_frame_tmp = np.loadtxt(
        base_path + "total_frame.txt").astype(np.int64)

    crop_width = crop_size_list[count, 0]
    crop_height = crop_size_list[count, 1]

    new_lineagefilename = new_lineagefile[count]  # + ".npy"

    analdir_name = "analysis/"
    analdir_path = base_path + analdir_name
    trackdata_cropdir_name = "trackdata_crop_w=%d_h=%d/" % (
        crop_width, crop_height)
    trackdata_cropdir_path = analdir_path + trackdata_cropdir_name

    dir_network = analdir_path + \
        "network%s_crop_w=%d_h=%d/" % (network_num, crop_width, crop_height)
    trackdata_cropdir_name_new = "network%s_trackdata_crop_w=%d_h=%d/" % (
        network_num, crop_width, crop_height)
    trackdata_cropdir_path_new = dir_network + trackdata_cropdir_name_new

    sutil.MakeDirs(trackdata_cropdir_path_new)

    lineage_path = base_path + new_lineagefilename

    lineage_data = np.load(lineage_path)

    if count == 0:

        new_data_size = lineage_data.shape[1]
        data_all_new = np.empty((0, lineage_data.shape[1]))

    frame_npy = lineage_data[:, 0].astype(np.int64)

    count += 1

    for frame in range(total_frame_tmp):

        data_frame_new = np.empty((0, lineage_data.shape[1]))

        # Index of npy files starts from 0
        index = np.where(frame_npy == frame)[0]

        lineage_data_target = lineage_data[index, :]
        ID_target = lineage_data_target[:, 3].astype(np.int64)

        filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
            crop_width,  crop_height, frame)
        filepath = trackdata_cropdir_path + filename
        data = np.loadtxt(filepath)

        data_ID = data[:, 3].astype(np.int64)

        for ID in data_ID:

            lineage_data_tmp = lineage_data_target[np.where(ID_target == ID)[
                0], :]
            voronoi_tmp = lineage_data_tmp[0, 6]
            if voronoi_tmp < voronoi_thr:  # if zero

                # Set zero for 'VoronoiArea','G1MarkerInVoronoiArea','ActinSegmentationArea','G1MarkerInActinSegmentationArea'
                lineage_data_tmp[0, 6:10] = 0
                # print(lineage_data_tmp)

            data_all_new = np.append(data_all_new, lineage_data_tmp, axis=0)

            data_frame_new = np.append(
                data_frame_new, lineage_data_tmp, axis=0)

        filename_new = "trackdata_crop_w=%d_h=%d_%d.txt" % (
            crop_width,  crop_height, frame)
        filepath_new = trackdata_cropdir_path_new + filename_new

        np.savetxt(filepath_new, data_frame_new)

        data_all = np.append(data_all, data, axis=0)


# print(data_all_new.shape)
# print(data_all.shape)


# %%
########### Normalization for the data with new Feature  ############
# Load trackdata "trackdata_crop_w=%d_h=%d_%d.txt" from the created network dir "network%s_trackdata_crop_w=%d_h=%d/"
# The data is zero-filled data for voronoi-area-zero cells. Loaded as data_all for all the dirs.
# Then, calculate min and max for each columns.

min_data_all_list = np.zeros(new_data_size)
max_data_all_list = np.zeros(new_data_size)
argmin_data_all_list = np.zeros(new_data_size).astype(np.int64)
argmax_data_all_list = np.zeros(new_data_size).astype(np.int64)


base_dir_list = sutil.ReadLinesText("base_path_list.txt")


data_all = np.empty((0, new_data_size))

for count, base_path in enumerate(base_dir_list):

    total_frame_tmp = np.loadtxt(
        base_path + "total_frame.txt").astype(np.int64)

    crop_width = crop_size_list[count, 0]
    crop_height = crop_size_list[count, 1]

    analdir_name = "analysis/"
    analdir_path = base_path + analdir_name

    dir_network = analdir_path + \
        "network%s_crop_w=%d_h=%d/" % (network_num, crop_width, crop_height)

    trackdata_cropdir_name = "network%s_trackdata_crop_w=%d_h=%d/" % (
        network_num, crop_width, crop_height)
    trackdata_cropdir_path = dir_network + trackdata_cropdir_name

    for frame in range(total_frame_tmp):
        filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
            crop_width,  crop_height, frame)
        filepath = trackdata_cropdir_path + filename
        data = np.loadtxt(filepath)
        # print(data.shape)
        data_all = np.append(data_all, data, axis=0)


for column_n in index_list_minmax:

    data_target = data_all[:, column_n]

    # if column_n == 18 or column_n == 19 or column_n == 20 or column_n == 21:   ## for dif1,2,3,4
    #data_target = data_target[np.where(data_target>=0)]

    min_data_all = np.min(data_target)
    max_data_all = np.max(data_target)

    argmin_data_all = np.argmin(data_target)
    argmax_data_all = np.argmax(data_target)

    min_data_all_list[column_n] = min_data_all
    max_data_all_list[column_n] = max_data_all

    argmin_data_all_list[column_n] = argmin_data_all
    argmax_data_all_list[column_n] = argmax_data_all


# %%
#  Save min and max values for each column

base_path_file = "base_path.txt"
base_path = sutil.Read1LineText(base_path_file)

crop_size_file = "crop_size.txt"
crop_size_array = np.loadtxt(crop_size_file).astype(np.int16)
crop_width = crop_size_array[0]
crop_height = crop_size_array[1]
crop_shift_height = crop_size_array[2]


analdir_name = "analysis/"
analdir_path = base_path + analdir_name

cellID_nlist_dir_name = "nlist_cellID_w=%d_h=%d/" % (crop_width, crop_height)
cellID_nlist_dir_path = analdir_path + cellID_nlist_dir_name


network_norm_path = analdir_path + \
    "network_info/network%s/crop%d_%d/" % (network_num,
                                           crop_width, crop_height)

sutil.MakeDirs(network_norm_path)

sutil.SaveListText(base_dir_list, network_norm_path + "min_max_dirlist.txt")


np.savetxt(network_norm_path + "data_all_vornoi0filled.txt", data_all)

col_count = 0
for column_n in index_list_minmax:

    # print(name_list_minmax[col_count])
    np.savetxt(network_norm_path + "min_max_%s.txt" % name_list_minmax[col_count], np.array(
        [min_data_all_list[column_n], max_data_all_list[column_n]]))
    col_count += 1


##### Save the cell property of cell which have min and max values for each column ####
col_count = 0
for column_n in index_list_minmax:

    argmin_index = argmin_data_all_list[column_n]
    argmax_index = argmax_data_all_list[column_n]

    print(name_list_minmax[col_count])
    print("min index = %d, value = %f" %
          (argmin_index, data_all[argmin_index][column_n]))
    print(data_all[argmin_index])
    print("max index = %d, value = %f" %
          (argmax_index, data_all[argmax_index][column_n]))
    print(data_all[argmax_index])
    np.savetxt(network_norm_path + "arg_min_%s.txt" %
               name_list_minmax[col_count], data_all[argmin_index])

    # print(name_list_minmax[col_count])
    np.savetxt(network_norm_path + "arg_max_%s.txt" %
               name_list_minmax[col_count], data_all[argmax_index])

    col_count += 1


# %%
# create network
# with time-reversal network


dir_network = analdir_path + \
    "network%s_crop_w=%d_h=%d/" % (network_num, crop_width, crop_height)

trackdata_cropdir_name = "network%s_trackdata_crop_w=%d_h=%d/" % (
    network_num, crop_width, crop_height)
trackdata_cropdir_path = dir_network + trackdata_cropdir_name


for shuffle in shuffle_list:
    for num_time in range(time_min, time_max+1):
        time_list = []

        for i in range(num_time):
            time_list.append("t"+str(i))

        networkdir_name = "network%s_num_w=%d_h=%d_time=%d/" % (
            network_num, crop_width, crop_height, num_time)
        networkdir_path = dir_network + networkdir_name

        sutil.MakeDirs(networkdir_path)

        childrendir_name = "children_w=%d_h=%d/" % (crop_width, crop_height)
        childrendir_path = analdir_path + childrendir_name

        parent_childrendir_name = "parent_children_w=%d_h=%d/" % (
            crop_width, crop_height)
        parent_childrendir_path = analdir_path + parent_childrendir_name

        for frame in range(total_frame-num_time):

            graph_data = {}

            for i in range(num_time):

                # Create network in the same time point

                trackfile_name = "trackdata_crop_w=%d_h=%d_%d.txt" % (
                    crop_width, crop_height, frame + i)
                track_load = np.loadtxt(
                    trackdata_cropdir_path + trackfile_name)

                CellID = track_load[:, 3].astype(np.int64)

                # Put cell nodes in the network

                nlist_name = "nlist_cellID_w=%d_h=%d_%d.txt" % (
                    crop_width, crop_height, frame + i)

                n_list = np.loadtxt(cellID_nlist_dir_path + nlist_name)

                src = n_list[:, 2]
                dst = n_list[:, 3]

                interaction_tuple = (time_list[i], 'interaction', time_list[i])
                graph_data[interaction_tuple] = (
                    th.tensor(src, dtype=th.int64), th.tensor(dst, dtype=th.int64))

                if i != num_time-1:

                    # Create network between two time points

                    parent_children_filename = "parent_children_%d.txt" % (
                        frame + i)
                    children_load = np.loadtxt(
                        parent_childrendir_path + parent_children_filename)

                    time_tuple = (time_list[i], 'time', time_list[i+1])

                    ### time-reversal network ####
                    time_tuple_rev = (time_list[i+1], 'time_rev', time_list[i])

                    src_time = children_load[:, 0].astype(np.int64)
                    dst_time = children_load[:, 1].astype(np.int64)

                    ### remove differentiation links in parent_childeren data####

                    trackfile_name2 = "trackdata_crop_w=%d_h=%d_%d.txt" % (
                        crop_width, crop_height, frame + i+1)
                    track_load2 = np.loadtxt(
                        trackdata_cropdir_path + trackfile_name2)

                    CellID2 = track_load2[:, 3].astype(np.int64)
                    dif_list = []
                    for index2 in CellID2:
                        if np.any(dst_time == index2) == True:
                            index_tmp = np.where(dst_time == index2)[0]
                            dif_list.append(index_tmp[0])

                    src_time = src_time[dif_list]
                    dst_time = dst_time[dif_list]

                    len_parent_children = len(src_time)

                    parent_children_new = np.zeros((len_parent_children, 2))

                    parent_children_new[:, 0] = src_time
                    parent_children_new[:, 1] = dst_time

                    parent_children_filename_new = "parent_children_new_%d.txt" % (
                        frame + i)
                    np.savetxt(parent_childrendir_path + parent_children_filename_new,
                               parent_children_new, header="CellID, ChildID")

                    graph_data[time_tuple] = (
                        th.tensor(src_time, dtype=th.int64), th.tensor(dst_time, dtype=th.int64))

                    ### time-reversal network ####
                    graph_data[time_tuple_rev] = (
                        th.tensor(dst_time, dtype=th.int64), th.tensor(src_time, dtype=th.int64))

            network_filename = "network_t=%dto%d.pickle" % (
                frame, frame+num_time-1)
            network_filepath = networkdir_path + network_filename

            sutil.PickleDump(graph_data, network_filepath)
            network_load = sutil.PickleLoad(network_filepath)


# %%
# import cell properties

n_labels_celltype_future = int(4)
n_labels_lineage = int(3)


for shuffle in shuffle_list:
    for num_time in range(time_min, time_max+1):
        time_list = []

        for i in range(num_time):
            time_list.append("t"+str(i))

        celltypedir_name = "celltype_w=%d_h=%d/" % (crop_width, crop_height)
        celltypedir_path = analdir_path + celltypedir_name

        networkdir_name = "network%s_num_w=%d_h=%d_time=%d/" % (
            network_num, crop_width, crop_height, num_time)
        networkdir_path = dir_network + networkdir_name

        shuffle_networkdir_name = "network%s_num_w=%d_h=%d_time=%d_shuffle/" % (
            network_num, crop_width, crop_height, num_time)
        shuffle_networkdir_path = dir_network+shuffle_networkdir_name

        sutil.MakeDirs(shuffle_networkdir_path)

        cellIDdir_name = "network%s_cellID_FinalLayer_num_w=%d_h=%d_time=%d/" % (
            network_num, crop_width, crop_height, num_time)
        cellIDdir_path = dir_network + cellIDdir_name

        sutil.MakeDirs(cellIDdir_path)

        shuffle_cellIDdir_name = "network%s_cellID_FinalLayer_num_w=%d_h=%d_time=%d_shuffle/" % (
            network_num, crop_width, crop_height, num_time)
        shuffle_cellIDdir_path = dir_network + shuffle_cellIDdir_name

        sutil.MakeDirs(shuffle_cellIDdir_path)

        cellIDdir_noborder_name = "network%s_cellID_FinalLayer_noborder_num_w=%d_h=%d_time=%d/" % (
            network_num, crop_width, crop_height, num_time)
        cellIDdir_noborder_path = dir_network + cellIDdir_noborder_name

        sutil.MakeDirs(cellIDdir_noborder_path)

        shuffle_cellIDdir_noborder_name = "network%s_cellID_FinalLayer_noborder_num_w=%d_h=%d_time=%d_shuffle/" % (
            network_num, crop_width, crop_height, num_time)
        shuffle_cellIDdir_noborder_path = dir_network + shuffle_cellIDdir_noborder_name

        sutil.MakeDirs(shuffle_cellIDdir_noborder_path)

        feature_histogram_dir_name = "network%s_FeatureHistogram_crop_w=%d_h=%d/" % (
            network_num, crop_width, crop_height)
        feature_histogram_dir_path = dir_network + feature_histogram_dir_name

        LineageTypedir_name = "network%s_LineageType_w=%d_h=%d_time=%d/" % (
            network_num, crop_width, crop_height, num_time)
        LineageTypedir_path = dir_network + LineageTypedir_name

        sutil.MakeDirs(LineageTypedir_path)

        noinfo_celltype = 3.0

        for frame in range(total_frame+1-num_time-1):
            network_filename = "network_t=%dto%d.pickle" % (
                frame, frame+num_time-1)
            network_filepath = networkdir_path + network_filename
            network_load = sutil.PickleLoad(network_filepath)

            g = dgl.heterograph(network_load)

            for i in range(num_time):

                #print("frame=%d, time=%d"%(frame,i))

                time_label = time_list[i]

                num_node = g.num_nodes(time_label)
                # print("num_node=%d"%num_node)

                trackfile_name = "trackdata_crop_w=%d_h=%d_%d.txt" % (
                    crop_width, crop_height, frame + i)
                track_load = np.loadtxt(
                    trackdata_cropdir_path + trackfile_name)

                CellID_withVoronoiZero = track_load[:, 3].astype(np.int64)
                X_withVoronoiZero = track_load[:, 4]
                Y_withVoronoiZero = track_load[:, 5]

                # load voronoi area and define voronoi_zero and nonzero cellIDx

                voronoi = track_load[:, 6]

                voronoi_zero_ID = track_load[np.where(
                    voronoi < voronoi_thr), 3].astype(np.int64)
                voronoi_nonzero_ID = track_load[np.where(
                    voronoi > voronoi_thr), 3].astype(np.int64)

                CellID_woVoronoiZero = track_load[np.where(
                    voronoi > voronoi_thr), 3].astype(np.int64)
                X_woVoronoiZero = track_load[np.where(
                    voronoi > voronoi_thr), 4]
                Y_woVoronoiZero = track_load[np.where(
                    voronoi > voronoi_thr), 5]

                # for 1st frame of  each network, we need to eliminate the div-dif cell because we don't have the cells in the network.

                if i == 0:
                    # remove voronoi = 0 cells
                    track_load = track_load[np.where(voronoi > voronoi_thr)]

                CellID = track_load[:, 3].astype(np.int64)

                X = track_load[:, 4]
                Y = track_load[:, 5]

                ######## Select bulk cells ##########

                col_count = 0
                for column_n in index_list_minmax:

                    temp_data = track_load[:, column_n]
                    temp_raw = np.zeros((num_node, 1))
                    temp_raw[CellID, 0] = temp_data
                    temp_norm_tmp = Normarization(
                        temp_data, min_data_all_list[column_n], max_data_all_list[column_n])
                    temp_norm = np.zeros((num_node, 1))  # Actinarea, G1actin
                    temp_norm[CellID, 0] = temp_norm_tmp

                    name_raw = name_list_minmax[col_count] + "_raw"
                    name_norm = name_list_minmax[col_count] + "_norm"
                    g.nodes[time_label].data[name_raw] = th.tensor(
                        temp_raw, dtype=th.float)
                    g.nodes[time_label].data[name_norm] = th.tensor(
                        temp_norm, dtype=th.float)

                    col_count += 1

                dum_one = np.ones((num_node, 1))  #

                # dummy zero
                dum_zero = np.zeros((num_node, 1))  #

                g.nodes[time_label].data['one'] = th.tensor(
                    dum_one, dtype=th.float)
                g.nodes[time_label].data['zero'] = th.tensor(
                    dum_zero, dtype=th.float)

                if i == num_time-1:  # for the last frame we need cell type

                    cellID_nlist_dir_name = "nlist_cellID_w=%d_h=%d/" % (
                        crop_width, crop_height)
                    cellID_nlist_dir_path = analdir_path + cellID_nlist_dir_name

                    label_nlist_dir_name = "neighbor_list/"
                    label_nlist_dir_path = analdir_path + label_nlist_dir_name

                    label_nlist = np.loadtxt(
                        label_nlist_dir_path + "neighbor_list_%d.txt" % (frame+i)).astype(np.int64)

                    tif_network_dir_name = "network%s_tif_overlap_network_w=%d_h=%d/time%d/" % (
                        network_num, crop_width, crop_height, num_time)
                    tif_network_dir_path = dir_network + tif_network_dir_name

                    sutil.MakeDirs(tif_network_dir_path)

                    tifdir_name = "tif/"
                    tifdir_path = analdir_path + tifdir_name

                    fig = plt.figure(figsize=(figx, figy))
                    ax = fig.add_subplot(1, 1, 1, aspect="equal")
                    fig.subplots_adjust(
                        left=left, right=right, bottom=bottom, top=top)

                    filename = "nlist_cellID_w=%d_h=%d_%d.txt" % (
                        crop_width, crop_height, frame+i)
                    nlist = np.loadtxt(cellID_nlist_dir_path + filename)

                    src_label = nlist[:, 0].astype(np.int64)
                    dst_label = nlist[:, 1].astype(np.int64)
                    src_ID = nlist[:, 2].astype(np.int64)
                    dst_ID = nlist[:, 3].astype(np.int64)

                    src_count = 0

                    border_index_list = []
                    for src in src_label:

                        label_list_s1 = np.where(label_nlist[:, 0] == src)[0]
                        label_list_s2 = np.where(label_nlist[:, 1] == src)[0]

                        n_label_s = len(label_list_s1) + len(label_list_s2)

                        ID_list_s = np.where(src_label == src)[0]

                        n_ID_s = len(ID_list_s)

                        if n_label_s != n_ID_s:

                            border_index_list.append(src_count)

                        src_count += 1

                    src_x = nlist[:, 4]
                    src_y = nlist[:, 5]

                    dst_x = nlist[:, 6]
                    dst_y = nlist[:, 7]

                    border_ID_list = np.unique(src_ID[border_index_list])

                    border_label_list = np.unique(src_label[border_index_list])

                    ################   remove border cells  ##########

                    # THIS is not used for plot. But Calculated for CellIDs for the final layer without voronoi area zero cells.
                    # we use CellID without voronoi area zero cells
                    CellID_noborder = CellID_woVoronoiZero.copy()
                    for border_ID in border_ID_list:
                        CellID_noborder = CellID_noborder[~(
                            CellID_noborder == border_ID)]

                    tifname = 'image_%d.tif' % (frame+i)
                    img_np = tifffile.imread(tifdir_path + tifname)

                    img_np[np.where(img_np > 0)] = 1

                    # print(np.shape(img_np))
                    width = np.shape(img_np)[1]
                    height = np.shape(img_np)[0]

                    xmin = (width - crop_width)/2
                    xmax = width - 1 - (width - crop_width)/2
                    ymin = (height - crop_height)/2 + crop_shift_height
                    ymax = height - 1 - \
                        (height - crop_height)/2 + crop_shift_height

                    for npair in range(len(src_x)):

                        ax.plot([src_x[npair], dst_x[npair]], [src_y[npair], dst_y[npair]], marker="o",
                                ms=1, color="yellow", linewidth=0.5)  # inverse xy axis to overlap image

                    for border_index in border_index_list:
                        ax.plot(src_x[border_index], src_y[border_index],
                                marker="o", ms=1, color="r")

                    # Plot voronoi zero cell

                    for zero_ID in voronoi_zero_ID[0]:
                        #print("zero voronoi area cells")
                        # print(zero_ID)
                        # print(X_withVoronoiZero[np.where(CellID_withVoronoiZero==zero_ID)])
                        # print(Y_withVoronoiZero[np.where(CellID_withVoronoiZero==zero_ID)])
                        ax.plot(X_withVoronoiZero[np.where(CellID_withVoronoiZero == zero_ID)], Y_withVoronoiZero[np.where(
                            CellID_withVoronoiZero == zero_ID)], marker="o", ms=1, color="b")

                    ax.set_xlim([0, width])
                    ax.set_ylim([height, 0])

                    ax.tick_params(bottom=False,
                                   left=False,
                                   right=False,
                                   top=False)

                    ax.plot([xmin, xmin], [ymin, ymax],
                            color="w", linewidth=0.5)
                    ax.plot([xmax, xmax], [ymin, ymax],
                            color="w", linewidth=0.5)
                    ax.plot([xmin, xmax], [ymin, ymin],
                            color="w", linewidth=0.5)
                    ax.plot([xmin, xmax], [ymax, ymax],
                            color="w", linewidth=0.5)

                    ax.imshow(img_np, cmap="Blues", alpha=0.5)

                    filename = 'tif_overlap_network_crop_w=%d_h=%d_%d.png' % (
                        crop_width,  crop_height, frame+i)
                    plt.savefig(tif_network_dir_path +
                                filename, format="png", dpi=300)

                    plt.close()

                    celltype_filename = "celltype_%d.txt" % (frame+i)
                    celltype_load = np.loadtxt(
                        celltypedir_path + celltype_filename).astype(np.int64)

                    # remove voronoi_area_zero cells for the first frame of each net
                    if i == 0:
                        cell_type_non_zero_index = []

                        for nonzero_ID in voronoi_nonzero_ID[0]:

                            cell_type_non_zero_index.append(
                                np.where(celltype_load[:, 0] == nonzero_ID)[0])

                    else:
                        cell_type_non_zero_index = range(
                            len(celltype_load[:, 0]))

                    celltype = np.zeros(num_node)

                    # cell state 3 for the last layer
                    celltype_noninfo = (np.ones((num_node, 1)))*noinfo_celltype

                    if shuffle == 0:
                        celltype[celltype_load[cell_type_non_zero_index, 0]
                                 ] = celltype_load[cell_type_non_zero_index, 1]

                    if shuffle == 1:

                        celltype[celltype_load[cell_type_non_zero_index, 0]] = np.random.permutation(
                            celltype_load[cell_type_non_zero_index, 1])

                    g.nodes[time_label].data['celltype'] = th.tensor(
                        celltype, dtype=th.int64)

                    g.nodes[time_label].data['celltype_future_int'] = th.tensor(
                        celltype_noninfo, dtype=th.int64)

                    g.nodes[time_label].data['celltype_future_norm'] = th.tensor(
                        celltype_noninfo/noinfo_celltype, dtype=th.float)

                    # for the last frame

                    celltype_future_onehot = IntArrayToOneHotVector(
                        celltype_noninfo, n_labels_celltype_future)

                    g.nodes[time_label].data['celltype_future_onehot'] = th.tensor(
                        celltype_future_onehot, dtype=th.float)

                    # Cell IDs at final layer without voronoi area zero cells
                    if shuffle == 0:
                        filename = "CellID_FinalLayer_t=%dto%d.txt" % (
                            frame+i-num_time+1, frame+i)
                        np.savetxt(cellIDdir_path + filename, CellID, fmt="%d")
                        np.savetxt(cellIDdir_noborder_path +
                                   filename, CellID_noborder, fmt="%d")

                    if shuffle == 1:
                        filename = "CellID_FinalLayer_t=%dto%d_shuffle.txt" % (
                            frame+i-num_time+1, frame+i)
                        np.savetxt(shuffle_cellIDdir_path +
                                   filename, CellID, fmt="%d")
                        np.savetxt(shuffle_cellIDdir_noborder_path +
                                   filename, CellID_noborder, fmt="%d")

                else:

                    celltype_filename = "celltype_%d.txt" % (frame+i)
                    celltype_load = np.loadtxt(
                        celltypedir_path + celltype_filename).astype(np.int64)

                    # remove voronoi_area_zero cells for the first frame of each net
                    if i == 0:
                        cell_type_non_zero_index = []

                        for nonzero_ID in voronoi_nonzero_ID[0]:

                            cell_type_non_zero_index.append(
                                np.where(celltype_load[:, 0] == nonzero_ID)[0])

                    else:
                        cell_type_non_zero_index = range(
                            len(celltype_load[:, 0]))

                    celltype = np.zeros((num_node, 1))

                    celltype[celltype_load[cell_type_non_zero_index, 0],
                             0] = celltype_load[cell_type_non_zero_index, 1]

                    g.nodes[time_label].data['celltype_future_int'] = th.tensor(
                        celltype, dtype=th.int64)

                    g.nodes[time_label].data['celltype_future_norm'] = th.tensor(
                        celltype/noinfo_celltype, dtype=th.float)

                    celltype_future_onehot = IntArrayToOneHotVector(
                        celltype, n_labels_celltype_future)

                    g.nodes[time_label].data['celltype_future_onehot'] = th.tensor(
                        celltype_future_onehot, dtype=th.float)

                # dummy zero
                LineageType_norm_array = np.zeros((num_node, 1))

                LineageType_int_array = np.zeros((num_node, 1))

                if i == 0:   # fill zero for Lineage Type of initial frame t0

                    g.nodes[time_label].data['lineage_norm'] = th.tensor(
                        LineageType_norm_array, dtype=th.float)

                    g.nodes[time_label].data['lineage_int'] = th.tensor(
                        LineageType_int_array, dtype=th.int64)

                    LineageType_onehot = IntArrayToOneHotVector(
                        LineageType_int_array, n_labels_lineage)
                    g.nodes[time_label].data['lineage_onehot'] = th.tensor(
                        LineageType_onehot, dtype=th.float)

                link_LineageType = []
                # Parental information
                if i != 0:

                    # Create network between two time points

                    parent_children_filename = "parent_children_new_%d.txt" % (
                        frame + i-1)  # we need -1
                    children_load = np.loadtxt(
                        parent_childrendir_path + parent_children_filename).astype(np.int64)

                    src_time = children_load[:, 0]
                    dst_time = children_load[:, 1]

                    for parent_index in src_time:

                        child_index = dst_time[np.where(
                            src_time == parent_index)]

                        for child in child_index:
                            link_LineageType.append([child, len(child_index)])

                            LineageType_int_array[child] = len(child_index)

                            # hv_LineageType_norm_array[child,2] = len(child_index)/2.0  # normalize lineage  how to be born. If born by division, 1.0. If no division, 0.5, If others, 0
                            LineageType_norm_array[child] = len(
                                child_index)/2.0

                    link_LineageType_nd = np.array(
                        link_LineageType).astype(np.int64)

                    g.nodes[time_label].data['lineage_norm'] = th.tensor(
                        LineageType_norm_array, dtype=th.float)

                    g.nodes[time_label].data['lineage_int'] = th.tensor(
                        LineageType_int_array, dtype=th.int64)

                    LineageType_onehot = IntArrayToOneHotVector(
                        LineageType_int_array, n_labels_lineage)

                    g.nodes[time_label].data['lineage_onehot'] = th.tensor(
                        LineageType_onehot, dtype=th.float)

                    # how to be born. If born by division, 2. If no division, 1, If others, 0
                    LineageType_filename = "LineageType_%d.txt" % (frame + i)

                    np.savetxt(LineageTypedir_path + LineageType_filename, link_LineageType_nd,
                               header="CellID, LineageType (div:2,none:1,others:0)")

            if shuffle == 0:
                filename = "NetworkWithFeartures_t=%dto%d.pickle" % (
                    frame, frame+num_time-1)

                sutil.PickleDump(g, networkdir_path + filename)

            if shuffle == 1:
                filename = "NetworkWithFeartures_t=%dto%d.pickle" % (
                    frame, frame+num_time-1)

                sutil.PickleDump(g, shuffle_networkdir_path + filename)

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

""" Load created graphs and make the NFB feature one-hot vecotors """

from functions import system_utility as sutil
from distutils.dir_util import copy_tree
import numpy as np
import os
import sys
version = 2
print("version=%d" % version)



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
######### Load parameters from yml file #########
filename_yml = 'input_create_network.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

exp_network = yaml_obj["network_name"]
time_min = yaml_obj["time_min"]
time_max = yaml_obj["time_max"]
shuffle_switch = yaml_obj["shuffle_switch"]
# exp_sim_switch = yaml_obj["exp_sim_switch"]# if 1, experiment. if 0, simulation

#tif_size = yaml_obj["tif_size"]


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
#### Define dir and file names ####

analdir_name = "analysis/"
analdir_path = base_path + analdir_name

dir_network = analdir_path + \
    "network%s_crop_w=%d_h=%d/" % (network_num, crop_width, crop_height)


new_network_num = network_num + suffix
new_dir_network = analdir_path + \
    "network%s_crop_w=%d_h=%d/" % (new_network_num, crop_width, crop_height)
sutil.MakeDirs(new_dir_network)

#trackdata_cropdir_name = "network%s_trackdata_crop_w=%d_h=%d/"%(network_num, crop_width,crop_height)
#trackdata_cropdir_path = dir_network  + trackdata_cropdir_name


# %%
#margin = 20
# for shuffle in [0,1]:
for shuffle in shuffle_list:
    for num_time in range(time_min, time_max+1):
        # for num_time in [4]:
        time_list = []

        for i in range(num_time):
            time_list.append("t"+str(i))

        celltypedir_name = "celltype_w=%d_h=%d/" % (crop_width, crop_height)
        celltypedir_path = analdir_path + celltypedir_name

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
                #save_graphs(networkdir_path + filename, g)

                g = sutil.PickleLoad(networkdir_path + filename)

            if shuffle == 1:
                filename = "NetworkWithFeartures_t=%dto%d.pickle" % (
                    frame, frame+num_time-1)
                #save_graphs(networkdir_path + filename, g)

                g = sutil.PickleLoad(shuffle_networkdir_path + filename)

            ##### Edit one-hot vector of F,L to reduce the dimension #######

            for i in range(num_time):

                time_label = time_list[i]

                g.nodes[time_label].data['lineage_onehot2'] = g.nodes[time_label].data['lineage_onehot'][:, 1:3]
                g.nodes[time_label].data['celltype_future_onehot2'] = g.nodes[time_label].data['celltype_future_onehot'][:, 0:3]

            ####### Save the network #######

            filename = "NetworkWithFeartures_t=%dto%d.pickle" % (
                frame, frame+num_time-1)
            #save_graphs(networkdir_path + filename, g)

            sutil.PickleDump(g, new_networkdir_path + filename)

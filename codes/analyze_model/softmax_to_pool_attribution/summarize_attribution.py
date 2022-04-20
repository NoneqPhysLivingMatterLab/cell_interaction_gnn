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

""" Summarize attributions """

from functions import system_utility as sutil 
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import collections
from training import gnn_models
import os
import sys
version = 8
print("version=%d" % version)

print("summarize attribution")



# %%
# Const
def_font_xtick = 9

# parameters for plot
filename_fig_yml = 'input_fig_config.yml'
path_fig_yml = "./" + filename_fig_yml
yaml_fig_obj = sutil.LoadYml(path_fig_yml)


figx = yaml_fig_obj["figx"]
figy = yaml_fig_obj["figy"]
left = yaml_fig_obj["left"]
right = yaml_fig_obj["right"]
bottom = yaml_fig_obj["bottom"]
top = yaml_fig_obj["top"]
dpi = yaml_fig_obj["dpi"]

fig_factor = yaml_fig_obj["fig_factor"]
legend_size = yaml_fig_obj["legend_size"]

msize = yaml_fig_obj["msize"]
capsize = yaml_fig_obj["capsize"]

axhline_lw = axis_width = yaml_fig_obj["axes.linewidth"]

lw = yaml_fig_obj["lw"]


plt.rcParams['font.family'] = yaml_fig_obj["font.family"]
plt.rcParams['xtick.direction'] = yaml_fig_obj["xtick.direction"]
plt.rcParams['ytick.direction'] = yaml_fig_obj["ytick.direction"]
plt.rcParams['xtick.major.width'] = yaml_fig_obj["xtick.major.width"]
plt.rcParams['ytick.major.width'] = yaml_fig_obj["ytick.major.width"]
plt.rcParams['font.size'] = yaml_fig_obj["font.size"]
plt.rcParams['axes.linewidth'] = yaml_fig_obj["axes.linewidth"]


# parameters
filename_yml = 'input_plot_attribution.yml'
path_yml = "./" + filename_yml
yaml_obj = sutil.LoadYml(path_yml)

n = yaml_obj["n"]


ModelType = yaml_obj["ModelType"]
celltype_list = yaml_obj["celltype_list"]
gpu = yaml_obj["gpu"]
n_net = yaml_obj["n_net"]
top_n = yaml_obj["top_n"]
LabelType = yaml_obj["LabelType"]
skip_misc = yaml_obj["skip_misc"]
m = yaml_obj["m"]  # log top m th attribution data for plot
m_plot = yaml_obj["m_plot"]  # plot only m_plotth


# %%
if gpu != -1:  # if gpu==-1, use cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu

print(th.cuda.device_count(), "GPUs available")
print(th.__version__)  # 0.4.0


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)
# print(th.cuda.current_device())


# %%
# print(ModelType)
basedir_filename = "base_path.txt"
base_dir = sutil.Read1LineText(basedir_filename) + "/"
# print(base_dir)


# network used in learning
networkdir_path = base_dir + "network/"
files = os.listdir(networkdir_path)
# print(files)

files_test = [s for s in files if 'test' in s]

if n_net == "all":
    n_net = len(files_test)
else:
    n_net = int(n_net)


att_test_dir = base_dir + "attribution_n=%d_%s/test/" % (n, ModelType)


if skip_misc == 0:

    fig_dir = base_dir + \
        "attribution_n=%d_%s_nnet=%d/test_summary/figs/" % (
            n, ModelType, n_net)
    data_dir = base_dir + \
        "attribution_n=%d_%s_nnet=%d/test_summary/data/" % (
            n, ModelType, n_net)


# %%
parameter_path = base_dir + "parameters.pickle"
param = sutil.PickleLoad(parameter_path)
hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature, reg, norm_final = gnn_models.LoadTrainingParameters(
    param)


# %%
# Load IG files and extract IGs of each cell in each subgraph.

num_time = len(time_list)

m_list = range(10)

top_list_all = []
abs_top_list_all = []


for CorrectClass in [0, 1, 2]:

    top_list = []
    abs_top_list = []

    for i in range(n_net):

        network_filepath = base_dir + "network/test_%d.pickle" % i
        network_load = sutil.PickleLoad(network_filepath)

        label_dir = att_test_dir + "test_%d/AllCells/%d/" % (i, CorrectClass)

        # print("label_dir")
        # print(label_dir)

        files_tmp = os.listdir(label_dir)

        # print(files_tmp)

        if LabelType != "all" and LabelType != "AllCells":
            files = [s for s in files_tmp if LabelType in s]
        else:
            files = files_tmp

        # print(files)

        frame_start = i
        frame_end = i+num_time-1

        network_path = networkdir_path + files_test[i]
        # print(network_path)
        network_load = sutil.PickleLoad(network_path)

        # print("files")
        # print(files)

        for target_dirname in files:

            cellIDdir_save = label_dir + target_dirname + "/"

            y_score_list_save = cellIDdir_save + "y_score_list.pickle"
            IntegratedGradient_list_save = cellIDdir_save + "IntegratedGradient_list.pickle"
            AllTargetCellListNoDoubling_save = cellIDdir_save + \
                "AllTargetCellListNoDoubling.pickle"
            AllNeighborCellList_save = cellIDdir_save + "AllNeighborCellList.pickle"
            LineageList_save = cellIDdir_save + "LineageList.pickle"

            y_score_list = sutil.PickleLoad(y_score_list_save)
            IntegratedGradient_list = sutil.PickleLoad(
                IntegratedGradient_list_save)
            AllTargetCellListNoDoubling = sutil.PickleLoad(
                AllTargetCellListNoDoubling_save)
            AllNeighborCellList = sutil.PickleLoad(AllNeighborCellList_save)
            LineageList = sutil.PickleLoad(LineageList_save)

            # Conversion of list IntegratedGradient_list to numpy IntegratedGradient_target
            IntegratedGradient_target = np.empty(
                (0, input_size), dtype=np.float32)

            features_target = np.empty((0, input_size), dtype=np.float32)

            # Save only IGs in Target subgraph in IntegratedGradient_target
            for j in range(num_time):

                time_name = "t"+"%d" % j
                IntegratedGradient_list_tmp = IntegratedGradient_list[j][AllTargetCellListNoDoubling[j]].to(
                    "cpu").detach().numpy().copy()
                IntegratedGradient_target = np.append(
                    IntegratedGradient_target, IntegratedGradient_list_tmp, axis=0)

                features_tmp = network_load.nodes[time_name].data[feature][AllTargetCellListNoDoubling[j]].to(
                    "cpu").detach().numpy().copy()
                features_target = np.append(
                    features_target, features_tmp, axis=0)

            # time, position number array corresponding to ntegratedGradient_target
            time_index_all = np.empty((0), int)
            position_all = np.empty((0), int)

            for j in range(num_time):

                if len(AllTargetCellListNoDoubling) != num_time:
                    print("Time points error")
                TargetCellList = AllTargetCellListNoDoubling[j]

                shape_array = len(TargetCellList)
                time_index_tmp = np.zeros(shape_array).astype(int)
                time_index_tmp[:] = j

                position_tmp = np.zeros(shape_array).astype(int)

                count = 0
                for tmp in TargetCellList:

                    if (len(LineageList) - (num_time-j)) >= 0:

                        # if the cell is in the main lineager, set 1
                        if tmp == LineageList[num_time-j-1]:
                            position_tmp[count] = 1

                    if (len(AllNeighborCellList) - (num_time-j)) >= 0:

                        count_neighbor = len(
                            AllNeighborCellList) - (num_time-j)

                        # if the cell is nighbor cells set 2
                        if (tmp in AllNeighborCellList[count_neighbor]) == True:
                            position_tmp[count] = 2

                    count += 1

                time_index_all = np.append(
                    time_index_all, time_index_tmp, axis=0)
                position_all = np.append(position_all, position_tmp, axis=0)

            IntegratedGradient_target_1D = IntegratedGradient_target.ravel()

            features_target_1D = features_target.ravel()

            # we take absolute value
            abs_IntegratedGradient_target_1D = np.abs(
                IntegratedGradient_target.ravel())

            ## sort ###
            value_list = np.sort(IntegratedGradient_target_1D)[::-1]
            index_list = np.argsort(IntegratedGradient_target_1D)[::-1]

            feature_index_list = index_list % input_size
            ID_index_list = index_list//input_size

            ## sort ###
            abs_value_list = np.sort(abs_IntegratedGradient_target_1D)[::-1]
            abs_index_list = np.argsort(abs_IntegratedGradient_target_1D)[::-1]

            abs_feature_index_list = abs_index_list % input_size
            abs_ID_index_list = abs_index_list//input_size

            IntegratedGradient_target_1D_path = cellIDdir_save + \
                "IntegratedGradient_target_1D.pickle"
            features_target_1D_path = cellIDdir_save + "features_target_1D.pickle"

            abs_IntegratedGradient_target_1D_path = cellIDdir_save + \
                "abs_IntegratedGradient_target_1D.pickle"

            value_list_path = cellIDdir_save + "attribution_sort_value_list.pickle"
            index_list_path = cellIDdir_save + "attribution_sort_1Dindex_list.pickle"
            feature_index_list_path = cellIDdir_save + \
                "attribution_sort_feature_index_list.pickle"
            ID_index_list_path = cellIDdir_save + "attribution_sort_ID_index_list.pickle"

            abs_value_list_path = cellIDdir_save + "abs_attribution_sort_value_list.pickle"
            abs_index_list_path = cellIDdir_save + "abs_attribution_sort_1Dindex_list.pickle"
            abs_feature_index_list_path = cellIDdir_save + \
                "abs_attribution_sort_feature_index_list.pickle"
            abs_ID_index_list_path = cellIDdir_save + \
                "abs_attribution_sort_ID_index_list.pickle"

            time_index_all_path = cellIDdir_save + "time_index_all.pickle"
            position_all_path = cellIDdir_save + "position_all_path.pickle"

            #print(IntegratedGradient_target_1D_path )
            sutil.PickleDump(IntegratedGradient_target_1D,
                             IntegratedGradient_target_1D_path)
            # print(IntegratedGradient_target_1D_path)

            sutil.PickleDump(features_target_1D, features_target_1D_path)

            sutil.PickleDump(abs_IntegratedGradient_target_1D,
                             abs_IntegratedGradient_target_1D_path)
            sutil.PickleDump(value_list, value_list_path)
            sutil.PickleDump(index_list, index_list_path)
            sutil.PickleDump(feature_index_list, feature_index_list_path)
            sutil.PickleDump(ID_index_list, ID_index_list_path)

            sutil.PickleDump(abs_value_list, abs_value_list_path)
            sutil.PickleDump(abs_index_list, abs_index_list_path)
            sutil.PickleDump(abs_feature_index_list,
                             abs_feature_index_list_path)
            sutil.PickleDump(abs_ID_index_list, abs_ID_index_list_path)

            sutil.PickleDump(time_index_all, time_index_all_path)
            sutil.PickleDump(position_all, position_all_path)

            value = value_list[m_list]
            index = index_list[m_list]

            ID_index = ID_index_list[m_list]
            feature_index = feature_index_list[m_list]

            time_type = time_index_all[ID_index]
            position_type = position_all[ID_index]

            top_list_each = [m_list, CorrectClass, time_type,
                             position_type, feature_index, value]
            # print(top_list_each)

            top_list.append(top_list_each)

            abs_value = abs_value_list[m_list]
            abs_index = abs_index_list[m_list]

            abs_ID_index = abs_ID_index_list[m_list]
            abs_feature_index = abs_feature_index_list[m_list]

            time_type = time_index_all[abs_ID_index]
            position_type = position_all[abs_ID_index]

            abs_top_list_each = [m_list, CorrectClass, time_type,
                                 position_type, abs_feature_index, abs_value]
            # print(top_list_each)

            abs_top_list.append(abs_top_list_each)

    top_list_all.append(top_list)

    abs_top_list_all.append(abs_top_list)


# %%
# We don't use the output below
if skip_misc == 0:

    top_list_path = data_dir + \
        "%s_top_attribution_list_m=%d.pickle" % (LabelType, m)
    sutil.PickleDump(top_list_all, top_list_path)

    abs_top_list_path = data_dir + \
        "%s_abs_top_attribution_list_m=%d.pickle" % (LabelType, m)
    sutil.PickleDump(abs_top_list_all, abs_top_list_path)


# %%
if skip_misc == 0:
    # top kth statistics

    print(base_dir)
    print("raw")

    title_fig = "Plot%dthFromTop%dthAttributionsForEach%sLabel" % (
        m_plot, m, LabelType)

    ratio = 2

    fig = plt.figure(figsize=(figx*3*ratio, figy*m_plot*ratio))

    fig.subplots_adjust(hspace=0.6, wspace=0.6)

    count_fig = 1

    for k in range(m_plot):
        top_name_list_all = []

        top_list_path = data_dir + \
            "%s_top%dth_attribution_list_name_m=%d.pickle" % (LabelType, k, m)

        for i in range(len(top_list_all)):

            ax = fig.add_subplot(m_plot, 3, count_fig)
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

            top_name_list = []

            top_list_tmp1 = top_list_all[i]

            for j in range(len(top_list_all[i])):

                top_list_tmp2 = top_list_tmp1[j]

                time_type = top_list_tmp2[2][k]
                position_type = top_list_tmp2[3][k]
                feature_type = top_list_tmp2[4][k]

                cell_type = "T%d\nP%d\nF%d" % (
                    time_type, position_type, feature_type)

                top_name_list.append(cell_type)

            # print("label=%d"%i)
            # print(top_name_list)

            top_name_list_all.append(top_name_list)

            # print("%dth"%k)
            # print(celltype_list[i])

            c = collections.Counter(top_name_list)

            most_common_name = c.most_common()
            # print(most_common_name)

            cell_type_name_list = []
            freq_list = []
            x_list = range(len(most_common_name))
            for l in x_list:
                cell_type_name = most_common_name[l][0]
                freq = most_common_name[l][1]

                cell_type_name_list.append(cell_type_name)
                freq_list.append(freq)

            title = "%sLabel=%sTop%dthAttribution" % (
                LabelType, celltype_list[i], k)
            ax.set_title(title)
            ax.bar(np.array(cell_type_name_list), np.array(freq_list))
            ax.set_xlabel("CellType")
            ax.set_ylabel("Frequency")
            fig.canvas.draw()
            xticklabels = ax.get_xticklabels()
            yticklabels = ax.get_yticklabels()

            if (len(x_list) > 0):
                font_xtick = def_font_xtick * (20/len(x_list))
            else:
                font_xtick = def_font_xtick
            if font_xtick >= def_font_xtick:
                font_xtick = def_font_xtick
            ax.set_xticklabels(xticklabels, fontsize=font_xtick)
            ax.set_yticklabels(yticklabels, fontsize=10)

            # plt.xticks(x_list,cell_type_name_list)

            count_fig += 1

            bar_x_filename = title + "_x.pickle"
            bar_y_filename = title + "_y.pickle"

            sutil.PickleDump(np.array(cell_type_name_list),
                             data_dir + bar_x_filename)
            sutil.PickleDump(np.array(freq_list), data_dir + bar_y_filename)

        # print(top_name_list_all)
        sutil.PickleDump(top_name_list_all, top_list_path)

    fig.show()

    filename_fig = title_fig + ".png"
    plt.savefig(fig_dir + filename_fig, format="png", dpi=dpi)
    filename_fig = title_fig + ".pdf"
    plt.savefig(fig_dir + filename_fig, format="pdf")

    # plt.show()
    # plt.close()


# %%

if skip_misc == 0:
    # top kth statistics

    print(base_dir)
    print("abs")
    import collections

    title_fig = "%s_abs_Plot%dthFromTop%dthAttributionsForEachCorrectLabel" % (
        LabelType, m_plot, m)

    ratio = 2

    fig = plt.figure(figsize=(figx*3*ratio, figy*m_plot*ratio))

    fig.subplots_adjust(hspace=0.6, wspace=0.6)

    count_fig = 1

    for k in range(m_plot):
        top_name_list_all = []

        top_list_path = data_dir + \
            "%s_abs_top%dth_attribution_list_name_m=%d.pickle" % (
                LabelType, k, m)

        for i in range(len(abs_top_list_all)):

            ax = fig.add_subplot(m_plot, 3, count_fig)
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

            top_name_list = []

            top_list_tmp1 = abs_top_list_all[i]

            for j in range(len(abs_top_list_all[i])):

                top_list_tmp2 = top_list_tmp1[j]

                time_type = top_list_tmp2[2][k]
                position_type = top_list_tmp2[3][k]
                feature_type = top_list_tmp2[4][k]

                cell_type = "T%d\nP%d\nF%d" % (
                    time_type, position_type, feature_type)

                top_name_list.append(cell_type)

            # print("label=%d"%i)
            # print(top_name_list)

            top_name_list_all.append(top_name_list)

            # print("%dth"%k)
            # print(celltype_list[i])

            c = collections.Counter(top_name_list)

            most_common_name = c.most_common()
            # print(most_common_name)

            cell_type_name_list = []
            freq_list = []
            x_list = range(len(most_common_name))
            for l in x_list:
                cell_type_name = most_common_name[l][0]
                freq = most_common_name[l][1]

                cell_type_name_list.append(cell_type_name)
                freq_list.append(freq)

            title = "%s_abs_CorrectLabel=%sTop%dthAttribution" % (
                LabelType, celltype_list[i], k)
            ax.set_title(title)
            ax.bar(np.array(cell_type_name_list), np.array(freq_list))
            ax.set_xlabel("CellType")
            ax.set_ylabel("Frequency")
            fig.canvas.draw()
            xticklabels = ax.get_xticklabels()
            yticklabels = ax.get_yticklabels()

            if (len(x_list) > 0):
                font_xtick = def_font_xtick * (20/len(x_list))
            else:
                font_xtick = def_font_xtick

            if font_xtick >= def_font_xtick:
                font_xtick = def_font_xtick
            ax.set_xticklabels(xticklabels, fontsize=font_xtick)
            ax.set_yticklabels(yticklabels, fontsize=10)

            # plt.xticks(x_list,cell_type_name_list)

            count_fig += 1

            bar_x_filename = title + "_x.pickle"
            bar_y_filename = title + "_y.pickle"

            sutil.PickleDump(np.array(cell_type_name_list),
                             data_dir + bar_x_filename)
            sutil.PickleDump(np.array(freq_list), data_dir + bar_y_filename)

        # print(top_name_list_all)
        sutil.PickleDump(top_name_list_all, top_list_path)

    fig.show()

    filename_fig = title_fig + ".png"
    plt.savefig(fig_dir + filename_fig, format="png", dpi=dpi)
    filename_fig = title_fig + ".pdf"
    plt.savefig(fig_dir + filename_fig, format="pdf")

    # plt.show()
    # plt.close()


# %%

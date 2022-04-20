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

""" Plot softmax scores """

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from training import gnn_models
from functions import system_utility as sutil
import os
import sys
version = 8
print("version=%d" % version)



celltype_list = ["No behavior", "Delamination", "Division"]

# %%
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

gpu = yaml_obj["gpu"]

#attribution_version = yaml_obj["attribution_version"]

n_net = yaml_obj["n_net"]
top_n = yaml_obj["top_n"]

LabelType = yaml_obj["LabelType"]

celltype_list = yaml_obj["celltype_list"]


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


networkdir_path = base_dir + "network/"
files = os.listdir(networkdir_path)


files_test = [s for s in files if 'test' in s]

if n_net == "all":
    n_net = len(files_test)
else:
    n_net = int(n_net)


att_test_dir = base_dir + "attribution_n=%d_%s/test/" % (n, ModelType)


fig_dir = base_dir + "attribution_n=%d_%s_nnet=%d/test_summary/softmax/figs/" % (
    n, ModelType, n_net)  # Changed the directory

sutil.MakeDirs(fig_dir)

data_dir = base_dir + "attribution_n=%d_%s_nnet=%d/test_summary/softmax/data/" % (
    n, ModelType, n_net)  # Changed the directory

sutil.MakeDirs(data_dir)


# %%
parameter_path = base_dir + "parameters.pickle"
param = sutil.PickleLoad(parameter_path)
hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature, reg, norm_final = gnn_models.LoadTrainingParameters(
    param)


# %%
LabelType_list = ["correct", "wrong", "all"]

# print(base_dir)
num_time = len(time_list)


for LabelType in LabelType_list:
    title_all = LabelType
    ratio = 1

    fig = plt.figure(figsize=(figx*3*ratio, figy*1*ratio))

    fig.subplots_adjust(hspace=1.2, wspace=0.4)

    #celltype_list = ["No behavior","Delamination","Division"]
    for CorrectClass in [0, 1, 2]:

        ax = fig.add_subplot(1, 3, CorrectClass+1)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

        data = []
        data_x = []

        for i in range(n_net):

            label_dir = att_test_dir + "test_%d/%d/" % (i, CorrectClass)

            files = os.listdir(label_dir)
            # print(files)

            frame_start = i
            frame_end = i+num_time-1

            network_path = networkdir_path + files_test[i]
            # print(network_path)
            network_load = sutil.PickleLoad(network_path)

            for target_dirname in files:

                if LabelType == "all":
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
                    AllNeighborCellList = sutil.PickleLoad(
                        AllNeighborCellList_save)
                    LineageList = sutil.PickleLoad(LineageList_save)

                    ax.plot(np.array(range(n+1))/n,
                            y_score_list, alpha=0.3, lw=lw)
                    data.append(y_score_list)
                    data_x.append(np.array(range(n+1))/n)

                elif LabelType in target_dirname:

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
                    AllNeighborCellList = sutil.PickleLoad(
                        AllNeighborCellList_save)
                    LineageList = sutil.PickleLoad(LineageList_save)

                    ax.plot(np.array(range(n+1))/n,
                            y_score_list, alpha=0.3, lw=lw)
                    data.append(y_score_list)
                    data_x.append(np.array(range(n+1))/n)

        title = "%s" % (celltype_list[CorrectClass])
        ax.set_title(title + "\n" + LabelType)
        xlabel = 'Alpha'
        ylabel = 'SoftmaxOutput_p'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim([-0.02, 1.02])

        sutil.PickleDump(data, data_dir + ylabel + "_" +
                         title + "_" + LabelType + "_y.pickle")  # Replaced
        sutil.PickleDump(data_x, data_dir + ylabel + "_" +
                         title + "_" + LabelType + "_x.pickle")  # Replaced

    #plt.savefig(cellIDdir_save + title + ".png",dpi=dpi)
    filename_fig = ylabel + "_" + title_all + ".png"
    plt.savefig(fig_dir + filename_fig, format="png", dpi=dpi)
    filename_fig = ylabel + "_" + title_all + ".pdf"
    plt.savefig(fig_dir + filename_fig, format="pdf")
    plt.show()
    plt.close()

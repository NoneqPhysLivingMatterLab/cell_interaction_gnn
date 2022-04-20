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

"""Pool IGs into cateories of features"""

'''
Option: With skip_misc_pool_attribution == 1, average, min, max IGs for each cell type of each target graph.
'''

import matplotlib.pyplot as plt
from functions import system_utility as sutil
import numpy as np
import torch as th
from training import gnn_models
import os
import sys
version = 8
print("version=%d" % version)



# %%
# Fixed parameters
clist = ["magenta", "cyan", "green", "blue", "black", "yellow"]
ratio = 1
plot = 0


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


# parameters for plot
filename_yml = 'input_plot_attribution.yml'
path_yml = "./" + filename_yml
yaml_obj = sutil.LoadYml(path_yml)

n = yaml_obj["n"]

#plot_option = yaml_obj["plot_option"]
# cell_select = yaml_obj["cell_select"] ## "correct" for only correct cells, all for all the cells

ModelType = yaml_obj["ModelType"]

#subnetwork_name = yaml_obj["subnetwork_name"]

gpu = yaml_obj["gpu"]

#attribution_version = yaml_obj["attribution_version"]

n_net = yaml_obj["n_net"]
top_n = yaml_obj["top_n"]

LabelType = yaml_obj["LabelType"]

skip_misc_pool_attribution = yaml_obj["skip_misc_pool_attribution"]

celltype_list = yaml_obj["celltype_list"]


# %%

if gpu != -1:  # if gpu==-1, use cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu

print(th.cuda.device_count(), "GPUs available")
print(th.__version__)  # 0.4.0


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)
print(th.cuda.current_device())


# %%
# print(ModelType)
top_index = range(top_n)


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


fig_dir = base_dir + \
    "attribution_n=%d_%s_nnet=%d/test_summary/%s/figs/" % (
        n, ModelType, n_net, LabelType)
data_dir = base_dir + \
    "attribution_n=%d_%s_nnet=%d/test_summary/%s/data/" % (
        n, ModelType, n_net, LabelType)


sutil.MakeDirs(data_dir)
sutil.MakeDirs(fig_dir)


# %%
parameter_path = base_dir + "parameters.pickle"
param = sutil.PickleLoad(parameter_path)
hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature, reg, norm_final = gnn_models.LoadTrainingParameters(
    param)


# %%
# Output cell type, IG,Time index,Position index of all the cells of all the graphs

""" Input and output of this cell
Input: 

- attribution_n=50_MaxMacroF1/test/test_0/AllCells/0/cellID=62_reallabel=2

    - time_index_all.pickle: see above. Output of summarize_attribution.py
    - position_all_path.pickle: see above. Output of summarize_attribution.py
    - IntegratedGradient_target_1D.pickle: see above. Output of summarize_attribution.py
    - features_target_1D.pickle: see above. Output of summarize_attribution.py

Output: 

- attribution_n=50_MaxMacroF1/test/test_0/AllCells/0/cellID=62_reallabel=2

    - cell_type_name_list.pickle: list of cell type name ("T%dP%dF%d"%(time_type,position_type,feature_type)) in the same order as IGs. 
    
        Using this cell type name list, we pool IGs in each subnetwork. 

    - ForEachCellType/data/T1P1F4

        IGs are pooled for each subgraph and for each feature category(ex.T1P1F4).

        - features.pickle
        - IG.pickle
        - index.pickle
    - misc: not used for the further analysis.
"""

num_time = len(time_list)

for CorrectClass in [0, 1, 2]:

    IntegratedGradient_target_1D_all = np.empty((0), np.float64)
    features_target_1D_all = np.empty((0), np.float64)
    time_index_1D_all = np.empty((0), int)
    position_1D_all = np.empty((0), int)

    cell_type_list_all = []

    for i in range(n_net):
        # print(i)

        label_dir = att_test_dir + "test_%d/AllCells/%d/" % (i, CorrectClass)

        files_tmp = os.listdir(label_dir)

        if LabelType != "all" and LabelType != "AllCells":
            files = [s for s in files_tmp if LabelType in s]
        else:
            files = files_tmp

        # print(files)

        frame_start = i
        frame_end = i+num_time-1

        for target_dirname in files:

            cellIDdir_save = label_dir + target_dirname + "/"

            Celltypedir_save = label_dir + target_dirname + "/" + "ForEachCellType/Fig/"
            sutil.MakeDirs(Celltypedir_save)

            Celltypedir_data_save = label_dir + target_dirname + "/" + "ForEachCellType/data/"
            sutil.MakeDirs(Celltypedir_data_save)

            IntegratedGradient_target_1D_path = cellIDdir_save + \
                "IntegratedGradient_target_1D.pickle"
            features_target_1D_path = cellIDdir_save + "features_target_1D.pickle"

            time_index_all_path = cellIDdir_save + "time_index_all.pickle"
            position_all_path = cellIDdir_save + "position_all_path.pickle"

            #print(IntegratedGradient_target_1D_path )
            IntegratedGradient_target_1D = sutil.PickleLoad(
                IntegratedGradient_target_1D_path)
            features_target_1D = sutil.PickleLoad(features_target_1D_path)
            # This is before sorting, and 1D
            time_index_all = sutil.PickleLoad(time_index_all_path)
            # This is before sorting, and 1D
            position_all = sutil.PickleLoad(position_all_path)

            size_1D = len(IntegratedGradient_target_1D)

            time_index_1D = np.zeros((size_1D), int)
            position_1D = np.zeros((size_1D), int)

            cell_type_list = []

            count = 0
            for time_index in time_index_all:
                # print(time_index)
                time_index_1D[count *
                              input_size:(count+1)*input_size] = time_index

                position_1D[count*input_size:(count+1)
                            * input_size] = position_all[count]

                count += 1

            for j in range(size_1D):
                time_type = time_index_1D[j]
                position_type = position_1D[j]
                feature_type = j % input_size
                cell_type = "T%dP%dF%d" % (
                    time_type, position_type, feature_type)
                cell_type_list.append(cell_type)

                cell_type_list_all.append(cell_type)

            #######################################################
            #######    Use IntegratedGradient_target_1D,cell_type_list  for each target cell in the final frame##########

            cell_type_name_path = cellIDdir_save + "cell_type_name_list.pickle"
            sutil.PickleDump(cell_type_list, cell_type_name_path)

            if plot == 1:
                fig = plt.figure(figsize=(figx*ratio*10, figy*ratio))
                ax = fig.add_subplot(1, 1, 1)
                fig.subplots_adjust(left=left, right=right,
                                    bottom=bottom, top=top)

            # print(Celltypedir_save)
            for j in range(num_time):
                for k in range(3):
                    for l in range(input_size):

                        cell_type = "T%dP%dF%d" % (j, k, l)
                        cell_type2 = "T%d\nP%d\nF%d" % (j, k, l)

                        EachCellDirPath = Celltypedir_data_save + cell_type + "/"
                        sutil.MakeDirs(EachCellDirPath)

                        index = [i for i, x in enumerate(
                            cell_type_list) if x == cell_type]
                        # print(index)

                        # print(cell_type)
                        IG = IntegratedGradient_target_1D[index]

                        features = features_target_1D[index]

                        IG_filename = "IG.pickle"
                        features_filename = "features.pickle"
                        index_filename = "index.pickle"

                        sutil.PickleDump(IG, EachCellDirPath + IG_filename)
                        sutil.PickleDump(
                            features, EachCellDirPath + features_filename)
                        sutil.PickleDump(
                            index, EachCellDirPath + index_filename)

                        xlist = [cell_type2]*len(index)

                        if plot == 1:
                            ax.set_title(target_dirname)
                            ax.scatter(xlist, IG, color=clist[j], s=10)

                        # ax2.set_title(target_dirname)
                        #ax2.scatter(xlist,IG,color=clist[j],s = 10)

            if plot == 1:
                fig.show()

                title_fig = "AttributionVsCellType"
                filename_fig = title_fig + ".png"
                plt.savefig(Celltypedir_save + filename_fig,
                            format="png", dpi=dpi)
                filename_fig = title_fig + ".pdf"
                plt.savefig(Celltypedir_save + filename_fig, format="pdf")

                # plt.show()
                plt.close()

            IntegratedGradient_target_1D_all = np.append(
                IntegratedGradient_target_1D_all, IntegratedGradient_target_1D, axis=0)
            features_target_1D_all = np.append(
                features_target_1D_all, features_target_1D, axis=0)
            time_index_1D_all = np.append(
                time_index_1D_all, time_index_1D, axis=0)
            position_1D_all = np.append(position_1D_all, position_1D, axis=0)

            #file_count += 1

    if skip_misc_pool_attribution == 1:
        # ex. the following files are saved in attribution_n=50_MaxMacroF1_nnet=11/test_summary/AllCells/data
        filename = "label=%d_%s_cell_type_list_all.pickle" % (
            CorrectClass, LabelType)
        sutil.PickleDump(cell_type_list_all, data_dir + filename)
        filename = "label=%d_%s_IntegratedGradient_target_1D_all.pickle" % (
            CorrectClass, LabelType)
        sutil.PickleDump(IntegratedGradient_target_1D_all, data_dir + filename)

        filename = "label=%d_%s_features_target_1D_all.pickle" % (
            CorrectClass, LabelType)
        sutil.PickleDump(features_target_1D_all, data_dir + filename)

        filename = "label=%d_%s_time_index_1D_all.pickle" % (
            CorrectClass, LabelType)
        sutil.PickleDump(time_index_1D_all, data_dir + filename)

        filename = "label=%d_%s_position_1D_all.pickle" % (
            CorrectClass, LabelType)
        sutil.PickleDump(position_1D_all, data_dir + filename)


# %%
# Load  IG.pickle, features.pickle of each subgraph and aggregate them into the IG, features for all the subgraph.

"""
Input: IGs pooled for each subgraph and for each feature category(ex.T1P1F4).

- attribution_n=50_MaxMacroF1/test/test_0/AllCells/0/cellID=62_reallabel=2
    - ForEachCellType/data/T1P1F4
        - features.pickle
        - IG.pickle
        - index.pickle
        
Output: 
- attribution_n=50_MaxMacroF1_nnet=11/test_summary/AllCells/data/0,1,2
    Pooled IG and feature values of all the subgraphs are aggregated from the above data for each subgraph.
    - 0/T1P1F2/
        - features_list_all.txt
        - IG_list_all.txt
    We use this data when we plot the pooled IGs.

"""


# print(base_dir)
#clist = make_cmlist2("viridis",num_time,200)

cellname_list = []

cellname_list2 = []
for j in range(num_time):
    for k in range(3):
        for l in range(input_size):

            cell_type = "T%dP%dF%d" % (j, k, l)
            cell_type2 = "T%d\nP%d\nF%d" % (j, k, l)

            cellname_list.append(cell_type)
            cellname_list2.append(cell_type2)


for CorrectClass in [0, 1, 2]:

    #print("label=%d"%CorrectClass )
    #fig = plt.figure(figsize=(figx,figy))

    for j in range(num_time):
        for k in range(3):
            for l in range(input_size):

                cell_type = "T%dP%dF%d" % (j, k, l)
                cell_type2 = "T%d\nP%d\nF%d" % (j, k, l)

                celltype_data_dir_save = data_dir + "%d/" % CorrectClass + cell_type + "/"
                sutil.MakeDirs(celltype_data_dir_save)

                IG_list_all_forEachType = np.empty((0))
                features_list_all_forEachType = np.empty((0))

                for i in range(n_net):
                    # print(i)

                    label_dir = att_test_dir + \
                        "test_%d/AllCells/%d/" % (i, CorrectClass)

                    files_tmp = os.listdir(label_dir)
                    # print(files)
                    if LabelType != "all" and LabelType != "AllCells":
                        files = [s for s in files_tmp if LabelType in s]
                    else:
                        files = files_tmp

                    frame_start = i
                    frame_end = i+num_time-1

                    for target_dirname in files:

                        #ax2 = fig2.add_subplot(file_total,1,file_count+1)

                        cellIDdir_save = label_dir + target_dirname + "/"

                        Celltypedir_save = label_dir + target_dirname + "/" + "ForEachCellType/Fig/"

                        Celltypedir_data_save = label_dir + target_dirname + "/" + "ForEachCellType/data/"

                        EachCellDirPath = Celltypedir_data_save + cell_type + "/"

                        IG_filename = "IG.pickle"
                        features_filename = "features.pickle"
                        index_filename = "index.pickle"

                        IG = sutil.PickleLoad(EachCellDirPath + IG_filename)
                        features = sutil.PickleLoad(
                            EachCellDirPath + features_filename)

                        # print(IG.shape)
                        # print(features.shape)
                        # print(features)

                        IG_list_all_forEachType = np.append(
                            IG_list_all_forEachType, IG)
                        features_list_all_forEachType = np.append(
                            features_list_all_forEachType, features)

                np.savetxt(celltype_data_dir_save +
                           "IG_list_all.txt", IG_list_all_forEachType)
                np.savetxt(celltype_data_dir_save +
                           "features_list_all.txt", features_list_all_forEachType)


# %%
# The following functions are used if skip_misc_pool_attribution == 1, when we want to average, min, max for each cell type of each target graph.

def IGs(cellname_list2, IG_max_average, IG_max_std, PoolType):
    # Plot IG_max
    fig = plt.figure(figsize=(figx*ratio*10, figy*ratio))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    # print(cellname_list2)
    # print(IG_max_average)
    ax.errorbar(cellname_list2, IG_max_average, yerr=IG_max_std, capsize=5,
                fmt='o', markersize=10, ecolor='black', markeredgecolor="black", color='w')
    ax.set_ylim([-np.nanmax(IG_max_average)*2.5,
                np.nanmax(IG_max_average)*2.5])

    ax.set_ylabel("Integrated gradient")
    ax.set_xlabel("Feature type")

    title_fig = "%s_AverageOf%sAttributionVsCellType" % (LabelType, PoolType)
    ax.set_title(title_fig+"_%d" % CorrectClass)

    filename_fig = title_fig + "_%d.png" % CorrectClass
    plt.savefig(fig_dir + filename_fig, format="png", dpi=dpi)
    filename_fig = title_fig + "_%d.pdf" % CorrectClass
    plt.savefig(fig_dir + filename_fig, format="pdf")

    # plt.show()
    plt.close()


def SortIGs(IG_max_average, cellname_list2, top_n, PoolType):
    IG_max_average_sort = np.sort(IG_max_average)[::-1]
    IG_max_average_sort_index = np.argsort(IG_max_average)[::-1]

    IG_max_average_sort_nonan = np.delete(
        IG_max_average_sort, np.where(np.isnan(IG_max_average_sort)), axis=0)
    IG_max_average_sort_index_nonan = np.delete(
        IG_max_average_sort_index, np.where(np.isnan(IG_max_average_sort)), axis=0)
    # print(np.where(np.isnan(IG_max_average_sort)))
    # print(IG_max_average_sort)
    # print(IG_max_average_sort_nonan)

    # print(IG_max_average_sort_index)

    IG_max_std_sort_nonan = IG_max_std[IG_max_average_sort_index_nonan]

    # print(cellname_list2)

    # print(cellname_list2[[0,1,2,3]])

    cellname_list2_sort = []

    # print(IG_max_average_sort_index)
    for index in IG_max_average_sort_index_nonan:
        cellname_list2_sort.append(cellname_list2[index])

    # print(cellname_list2_sort)
    fig2 = plt.figure(figsize=(figx*ratio, figy*ratio))
    ax2 = fig2.add_subplot(1, 1, 1)
    fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    ax2.errorbar(cellname_list2_sort[0:top_n], IG_max_average_sort_nonan[top_index], yerr=IG_max_std_sort_nonan[top_index],
                 capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor="black", color='w')
    # ax.set_ylim([-np.nanmax(IG_max_average_sort[top_index])*2.5,np.nanmax(IG_max_average_sort[top_index])*2.5])

    sutil.PickleDump(IG_max_std_sort_nonan, data_dir +
                     "label=%d_IG_%s_std_list_sort_nonan.pickle" % (CorrectClass, PoolType))

    sutil.PickleDump(cellname_list2_sort, data_dir +
                     "label=%d_cell_name_list_sort_nonan.pickle" % CorrectClass)

    sutil.PickleDump(IG_max_average_sort_nonan, data_dir +
                     "label=%d_IG_%s_average_list_sort_nonan.pickle" % (CorrectClass, PoolType))

    ax2.set_ylabel("Integrated gradient")
    ax2.set_xlabel("Feature type")

    title_fig = "AverageOf%sAttributionVsCellType_sort" % PoolType
    ax2.set_title(title_fig+"_%d" % CorrectClass)
    filename_fig = title_fig + "_%d.png" % CorrectClass
    plt.savefig(fig_dir + filename_fig, format="png", dpi=dpi)
    filename_fig = title_fig + "_%d.pdf" % CorrectClass
    plt.savefig(fig_dir + filename_fig, format="pdf")

    # plt.show()
    plt.close()

# %%
# Load  IG.pickle, features.pickle and ,average, min, max for each cell type of each target graph.
# We don't use the output from this cell when we plot the IGs as the bar plot.


if skip_misc_pool_attribution == 0:

    # print(base_dir)

    cellname_list = []

    cellname_list2 = []
    for j in range(num_time):
        for k in range(3):
            for l in range(input_size):

                cell_type = "T%dP%dF%d" % (j, k, l)
                cell_type2 = "T%d\nP%d\nF%d" % (j, k, l)

                cellname_list.append(cell_type)
                cellname_list2.append(cell_type2)

    for CorrectClass in [0, 1, 2]:

        #print("label=%d"%CorrectClass )
        #fig = plt.figure(figsize=(figx,figy))

        IG_max_list_all = np.empty((0, num_time*3*input_size))

        IG_mean_list_all = np.empty(
            (0, num_time*3*input_size))  # Added 20210412

        IG_min_list_all = np.empty(
            (0, num_time*3*input_size))  # Added 20210417

        # print(IG_max_list_all.shape)

        for i in range(n_net):
            # print(i)

            label_dir = att_test_dir + \
                "test_%d/AllCells/%d/" % (i, CorrectClass)

            files_tmp = os.listdir(label_dir)
            # print(files)
            if LabelType != "all" and LabelType != "AllCells":
                files = [s for s in files_tmp if LabelType in s]
            else:
                files = files_tmp

            frame_start = i
            frame_end = i+num_time-1

            for target_dirname in files:

                #ax2 = fig2.add_subplot(file_total,1,file_count+1)

                cellIDdir_save = label_dir + target_dirname + "/"

                Celltypedir_save = label_dir + target_dirname + "/" + "ForEachCellType/Fig/"

                Celltypedir_data_save = label_dir + target_dirname + "/" + "ForEachCellType/data/"

                # print(Celltypedir_save)

                IG_max_list = np.zeros((1, num_time*3*input_size))

                IG_mean_list = np.zeros(
                    (1, num_time*3*input_size))  # Added 20210412

                IG_min_list = np.zeros(
                    (1, num_time*3*input_size))  # Added 20210417

                count = 0
                for j in range(num_time):
                    for k in range(3):
                        for l in range(input_size):

                            cell_type = "T%dP%dF%d" % (j, k, l)
                            cell_type2 = "T%d\nP%d\nF%d" % (j, k, l)

                            EachCellDirPath = Celltypedir_data_save + cell_type + "/"

                            IG_filename = "IG.pickle"
                            features_filename = "features.pickle"
                            index_filename = "index.pickle"

                            IG = sutil.PickleLoad(
                                EachCellDirPath + IG_filename)
                            features = sutil.PickleLoad(
                                EachCellDirPath + features_filename)

                            if len(IG) >= 1:
                                IG_max = np.max(IG)
                                IG_mean = np.mean(IG)  # Added 20210412

                                IG_min = np.min(IG)  # Added 20210417

                            else:  # そのcell typeの細胞がいない時はnanをいれる。
                                IG_max = np.nan
                                IG_mean = np.nan
                                IG_min = np.nan

                            IG_max_list[0][count] = IG_max

                            IG_mean_list[0][count] = IG_mean  # Added 20210412
                            IG_min_list[0][count] = IG_min  # Added 20210417

                            count += 1

                #file_count += 1

                # print(IG_max_list)

                IG_max_list_all = np.append(
                    IG_max_list_all, IG_max_list, axis=0)

                IG_mean_list_all = np.append(
                    IG_mean_list_all, IG_mean_list, axis=0)  # Added 20210412

                IG_min_list_all = np.append(
                    IG_min_list_all, IG_min_list, axis=0)  # Added 20210417

        # print(IG_max_list_all.shape)

        sutil.PickleDump(cellname_list, data_dir + "cell_name_list.pickle")

        # print(IG_max_average)

        PoolType = "Max"
        sutil.PickleDump(IG_max_list_all, data_dir +
                         "label=%d_IG_%s_list.pickle" % (CorrectClass, PoolType))
        IG_max_average = np.nanmean(IG_max_list_all, axis=0)
        IG_max_std = np.nanstd(IG_max_list_all, axis=0, ddof=1)
        # print(IG_max_average)
        sutil.PickleDump(IG_max_average, data_dir +
                         "label=%d_IG_%s_average_list.pickle" % (CorrectClass, PoolType))
        sutil.PickleDump(IG_max_std, data_dir +
                         "label=%d_IG_%s_std_list.pickle" % (CorrectClass, PoolType))

        IGs(cellname_list2, IG_max_average, IG_max_std, PoolType)
        SortIGs(IG_max_average, cellname_list2, top_n, PoolType)

        PoolType = "Mean"
        sutil.PickleDump(IG_mean_list_all, data_dir +
                         "label=%d_IG_%s_list.pickle" % (CorrectClass, PoolType))
        IG_mean_average = np.nanmean(IG_mean_list_all, axis=0)
        IG_mean_std = np.nanstd(IG_mean_list_all, axis=0, ddof=1)
        sutil.PickleDump(IG_mean_average, data_dir +
                         "label=%d_IG_%s_average_list.pickle" % (CorrectClass, PoolType))
        sutil.PickleDump(IG_mean_std, data_dir +
                         "label=%d_IG_%s_std_list.pickle" % (CorrectClass, PoolType))

        IGs(cellname_list2, IG_mean_average, IG_mean_std, PoolType)
        SortIGs(IG_mean_average, cellname_list2, top_n, PoolType)

        PoolType = "Min"
        sutil.PickleDump(IG_min_list_all, data_dir +
                         "label=%d_IG_%s_list.pickle" % (CorrectClass, PoolType))
        IG_min_average = np.nanmean(IG_min_list_all, axis=0)
        IG_min_std = np.nanstd(IG_min_list_all, axis=0, ddof=1)
        sutil.PickleDump(IG_min_average, data_dir +
                         "label=%d_IG_%s_average_list.pickle" % (CorrectClass, PoolType))
        sutil.PickleDump(IG_min_std, data_dir +
                         "label=%d_IG_%s_std_list.pickle" % (CorrectClass, PoolType))
        IGs(cellname_list2, IG_min_average, IG_min_std, PoolType)
        SortIGs(IG_min_average, cellname_list2, top_n, PoolType)


# %%

# %%

print("finish all")

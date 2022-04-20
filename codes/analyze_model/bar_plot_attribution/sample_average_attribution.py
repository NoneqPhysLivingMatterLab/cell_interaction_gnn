# -*- coding: utf-8 -*-
# +
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

""" Sample-average attribution """

import time
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from training import gnn_models
from functions import system_utility as sutil
import os
import sys
version = 15
print("version=%d" % version)



# +
ratio = 1.5
# Load figure congiguration

filename_fig_yml = 'input_fig_config.yml'
path_fig_yml = "./" + filename_fig_yml
yml_fig_obj = sutil.LoadYml(path_fig_yml)


# Const
celltype_name_list = yml_fig_obj["celltype_name_list"]
gpu = yml_fig_obj["gpu"]

clist = yml_fig_obj["clist"]

color_list = yml_fig_obj["color_list"]


celltype_list = yml_fig_obj["celltype_list"]

figx_factor = yml_fig_obj["figx_factor"]  # def 1.3
figy_factor = yml_fig_obj["figy_factor"]  # def 13.2
figy_factor_wolabel = yml_fig_obj["figy_factor_wolabel"]

font_normal = yml_fig_obj["font_normal"]

nbin_y = yml_fig_obj["nbin_y"]

hspace = yml_fig_obj["hspace"]  # 0.2 for ext, 0.1for full

axhline_lw = yml_fig_obj["axhline_lw"]

axis_width = yml_fig_obj["axis_width"]

figx = yml_fig_obj["figx"]  # 3.14 for Full model, 2 for ext
figy = yml_fig_obj["figy"]  # 2.9 for Full model, 2 for ext
left = yml_fig_obj["left"]
right = yml_fig_obj["right"]
bottom = yml_fig_obj["bottom"]
top = yml_fig_obj["top"]
dpi = yml_fig_obj["dpi"]

capsize = yml_fig_obj["capsize"]


pad = yml_fig_obj["pad"]

axhline_lw = axis_width

locator_params_nbins = yml_fig_obj['locator_params_nbins']

# parameters for plot
plt.rcParams['font.family'] = yml_fig_obj['font.family']
plt.rcParams['xtick.direction'] = yml_fig_obj['xtick.direction']
plt.rcParams['ytick.direction'] = yml_fig_obj['ytick.direction']
plt.rcParams['xtick.major.width'] = axis_width
plt.rcParams['ytick.major.width'] = axis_width
plt.rcParams['font.size'] = yml_fig_obj['font.size']
plt.rcParams['axes.linewidth'] = axis_width
# plt.locator_params(axis='x',nbins=20)
plt.locator_params(axis='y', nbins=locator_params_nbins)
plt.rcParams["xtick.major.size"] = yml_fig_obj['xtick.major.size']
plt.rcParams["ytick.major.size"] = yml_fig_obj['ytick.major.size']

ymax_set = yml_fig_obj['ymax_set']
ymin_set = yml_fig_obj['ymin_set']


# +

if gpu != -1:  # if gpu==-1, use cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu

print(th.cuda.device_count(), "GPUs available")
print(th.__version__)  # 0.4.0


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)
print(th.cuda.current_device())


# -

def SortWoNan(average_mean):
    average_mean_sort = np.sort(average_mean)[::-1]
    average_mean_sort_index = np.argsort(average_mean)[::-1]

    average_mean_sort_nonan = np.delete(
        average_mean_sort, np.where(np.isnan(average_mean_sort)), axis=0)
    average_mean_sort_index_nonan = np.delete(
        average_mean_sort_index, np.where(np.isnan(average_mean_sort)), axis=0)

    return(average_mean_sort_nonan, average_mean_sort_index_nonan)


# +
# parameters for plot
filename_yml = 'input_bar_plot_attribution.yml'
path_yml = "./" + filename_yml
yaml_obj = sutil.LoadYml(path_yml)

n = yaml_obj["n"]


ModelType = yaml_obj["ModelType"]

n_net = yaml_obj["n_net"]
top_n = yaml_obj["top_n"]

LabelType = yaml_obj["LabelType"]

# max or mean, min  MeanForEachFutureFate ,MeanForAllCells
anal_type = yaml_obj["anal_type"]
Normalize = yaml_obj["Normalize"]  # RMS, Raw

attribution_dir_name_option = yaml_obj["attribution_dir_name_option"]

feature_list = yaml_obj["feature_list"]

save_dir_name = yaml_obj["save_dir_name"]


# +
# Load traget sample directory paths

# print(ModelType)

sample_list_load = sutil.ReadLinesText("base_path_list.txt")

sample_list_load = sutil.RemoveCyberduckPrefixSuffix(sample_list_load)


# +
# Create sample path list in which the attribution was correctly calculated. We use these paths for the following calculation.

sample_list = []
for i in sample_list_load:

    ###########Load data from each sample #############
    base_dir = i + "/"
    # print(base_dir)
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
    ###########Load data from each sample #############
    # Just to check if the attribution is correctly calculated, we search the following filename.
    pickle_path = base_dir + "attribution%s_n=%d_%s_nnet=%d/test_summary/%s/data/label=2_AllCells_IntegratedGradient_target_1D_all.pickle" % (
        attribution_dir_name_option, n, ModelType, n_net, LabelType)

    # print(pickle_path)

    if os.path.isfile(pickle_path) == True:
        sample_list.append(i)

# print(sample_list)
sample_n = len(sample_list)
# print(sample_n)

# +
# Calculate the average of IG values for each sample. Later we calculate the sample average.
# IG_average_list, rms_list2: (len(sample_list),num_time*3*input_size)) Averaged IGs for each cell fate labels.
# average_all: [IG_average_list for label0,IG_average_list for label1,IG_average_list for label2]

average_all = []

rms_all = []
for CorrectClass in [0, 1, 2]:

    #count_sample = 0
    for count_sample, i in enumerate(sample_list):

        ###########Load data from each sample #############
        base_dir = i+"/"
        # network used in learning
        networkdir_path = base_dir + "network/"
        files = os.listdir(networkdir_path)
        # print(files)

        files_test = [s for s in files if 'test' in s]

        if n_net == "all":
            n_net = len(files_test)
        else:
            n_net = int(n_net)

        parameter_path = base_dir + "parameters.pickle"

        data_dir = base_dir + "attribution%s_n=%d_%s_nnet=%d/test_summary/%s/data/%d/" % (
            attribution_dir_name_option, n, ModelType, n_net, LabelType, CorrectClass)

        param = sutil.PickleLoad(parameter_path)

        hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature, reg, norm_final = gnn_models.LoadTrainingParameters(
            param)

        num_time = len(time_list)

        #print("label=%d"%CorrectClass )

        if count_sample == 0:
            # IG_max_average_list = np.zeros((len(sample_list),num_time*3*input_size)) # The name was misleading. Changed the name.
            IG_average_list = np.zeros(
                (len(sample_list), num_time*3*input_size))

        if count_sample == 0:
            rms_list2 = np.zeros((len(sample_list), num_time*3*input_size))

        IG_list = np.zeros(0)
        rms_list = np.zeros(0)
        for j in range(num_time):
            for k in range(3):
                for l in range(input_size):

                    cell_type = "T%dP%dF%d" % (j, k, l)
                    cell_type2 = "T%d\nP%d\nF%d" % (j, k, l)

                    time = j-num_time + 1
                    feature_type = "%s of %s at t=%d" % (
                        feature_list[l], celltype_name_list[k], time)
                    # print(feature_type)

                    features_list_all_fs = os.path.getsize(
                        data_dir + cell_type + "/features_list_all.txt")
                    if features_list_all_fs != 0:

                        features_list_all = np.loadtxt(
                            data_dir + cell_type + "/features_list_all.txt")
                    else:  # If the feature is absent in the subgraph
                        features_list_all = []

                    IG_list_all_fs = os.path.getsize(
                        data_dir + cell_type + "/IG_list_all.txt")
                    if IG_list_all_fs != 0:

                        IG_list_all = np.loadtxt(
                            data_dir + cell_type + "/IG_list_all.txt")
                    else:  # If the feature is absent in the subgraph
                        IG_list_all = []

                    # If the feature is absent in the subgraph, set nan. For example, P=0 is not defined in the paper 2021. So np.nan is applied to such data.
                    if len(features_list_all) == 0:

                        tmp = np.nan
                        rms_tmp = np.nan

                    else:
                        if anal_type == "MeanForEachFutureFate":  # For NFB, we avearge only among the cells with the relavent NFB
                            # In the final frame(j=num_time-1)、since NFB=0, average of IG is set to zero by tmp=0.
                            if ((feature_list[l] == "NB") or (feature_list[l] == "Div") or (feature_list[l] == "Del")) and (j != num_time-1):

                                if (NoSelfInfo == 1) and (k == 1):  # Self-fearute of No self model
                                    tmp = 0
                                    rms_tmp = 0
                                else:
                                    # Weights option can be used only for the feature sets including the non-zero feature. If all are zero, we process by else.
                                    if np.max(features_list_all) > 0:
                                        tmp = np.average(
                                            IG_list_all, weights=features_list_all)
                                        rms_tmp = np.sqrt(np.average(
                                            IG_list_all**2, weights=features_list_all))

                                    else:
                                        tmp = 0
                                        rms_tmp = 0

                            else:  # For features except NFB, we avearge among all the cells
                                tmp = np.mean(IG_list_all)
                                rms_tmp = np.sqrt(np.mean(IG_list_all**2))

                        elif anal_type == "MeanForAllCells":
                            tmp = np.mean(IG_list_all)
                            rms_tmp = np.sqrt(np.mean(IG_list_all**2))

                    IG_list = np.append(IG_list, tmp)
                    rms_list = np.append(rms_list, rms_tmp)

        IG_average_list[count_sample][:] = IG_list
        rms_list2[count_sample][:] = rms_list

    average_all.append(IG_average_list)
    rms_all.append(rms_list2)


# +
# If we normalize the IG by "RMS", we run this cell.

average_all_cp = np.copy(average_all)
if Normalize == "RMS":
    for CorrectClass in [0, 1, 2]:

        for i, sample in enumerate(sample_list):
            for j in range(num_time):
                for k in range(3):
                    for l in range(input_size):

                        index_tmp = j*(3*input_size) + k*input_size + l
                        random_feature_index = j * \
                            (3*input_size) + k*input_size + (input_size-1)

                        if (NoSelfInfo == 1) and (k == 1):  # Self-fearute of No self model
                            average_all[CorrectClass][i][index_tmp] = 0
                        elif np.isnan(rms_all[CorrectClass][i][random_feature_index]) == False:
                            if rms_all[CorrectClass][i][random_feature_index] > 0:
                                average_all[CorrectClass][i][index_tmp] = average_all_cp[CorrectClass][i][index_tmp] / \
                                    rms_all[CorrectClass][i][random_feature_index]
                            else:
                                average_all[CorrectClass][i][index_tmp] = 0
                        else:
                            average_all[CorrectClass][i][index_tmp] = np.nan


# +
# Calculate the sample average and standard error from average_all.
# IG_average_list, rms_list2: (len(sample_list),num_time*3*input_size)) Averaged IGs for each cell fate labels.
# average_all: [IG_average_list for label0,IG_average_list for label1,IG_average_list for label2]
# The order of pooled cell types are listed in cellname_list.append(cell_type), cellname_list2.append(cell_type2),featurename_list.append(feature_type) in different naming formats


save_dir = "../bar_plot_result/" + save_dir_name + \
    "_sample=%d" % (sample_n) + "/%s/%s/%s/%s/" % (anal_type,
                                                   ModelType, LabelType, Normalize)
# print(save_dir)
sutil.MakeDirs(save_dir)

data_dir = save_dir + "data/"
fig_dir = save_dir + "fig/"

sutil.MakeDirs(data_dir)
sutil.MakeDirs(fig_dir)


sample_list_save = "sample_list_succeed.txt"
sutil.SaveListText(sample_list, data_dir + sample_list_save)


cellname_list = []

cellname_list2 = []

featurename_list = []

for j in range(num_time):
    for k in range(3):
        for l in range(input_size):

            cell_type = "T%dP%dF%d" % (j, k, l)
            cell_type2 = "T%d\nP%d\nF%d" % (j, k, l)

            time = j-num_time + 1
            feature_type = "%s of %s at t=%d" % (
                feature_list[l], celltype_name_list[k], time)
            # print(feature_type)

            cellname_list.append(cell_type)
            cellname_list2.append(cell_type2)
            featurename_list.append(feature_type)


for CorrectClass in [0, 1, 2]:
    average_list_target = average_all[CorrectClass]

    # print(average_list_target)
    # For the absent feautres (the IG is np.nan), RuntimeWarning: Degrees of freedom <= 0 for slice appear. But you can't ignore it.
    average_mean = np.nanmean(average_list_target, axis=0)
    # print(average_mean)

    average_std = np.nanstd(average_list_target, axis=0, ddof=1)
    # print(average_std)

    average_se = average_std / np.sqrt(len(sample_list))

    sutil.PickleDump(average_mean, data_dir +
                     "label=%d_average_mean.pickle" % CorrectClass)

    sutil.PickleDump(cellname_list, data_dir +
                     "label=%d_cellname_list.pickle" % CorrectClass)

    sutil.PickleDump(cellname_list2, data_dir +
                     "label=%d_cellname_list2.pickle" % CorrectClass)

    sutil.PickleDump(featurename_list, data_dir +
                     "label=%d_featurename_list.pickle" % CorrectClass)

    sutil.PickleDump(average_se, data_dir +
                     "label=%d_average_se.pickle" % CorrectClass)

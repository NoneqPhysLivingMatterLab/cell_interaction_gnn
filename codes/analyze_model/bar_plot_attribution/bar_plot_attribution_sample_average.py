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

"""Bar plot the sample-averaged attribution """

import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from training import gnn_models
from functions import system_utility as sutil
import os
import sys
version = 1
print("version=%d" % version)


# +
# Plot Max and Mean IG taken for each subnetwork


ratio = 1.5
filename_fig_yml = 'input_fig_config.yml'
path_fig_yml = "./" + filename_fig_yml
yml_fig_obj = sutil.LoadYml(path_fig_yml)


# Const
celltype_name_list = yml_fig_obj["celltype_name_list"]
gpu = yml_fig_obj["gpu"]

clist = yml_fig_obj["clist"]

color_list = yml_fig_obj["color_list"]


celltype_list = yml_fig_obj["celltype_list"]
celltype_label = yml_fig_obj["celltype_label"]


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
# print(th.cuda.current_device())

# +
# parameters for plot
filename_yml = 'input_bar_plot_attribution.yml'
path_yml = "./" + filename_yml
yaml_obj = sutil.LoadYml(path_yml)

n = yaml_obj["n"]


ModelType = yaml_obj["ModelType"]

#attribution_version = yaml_obj["attribution_version"]

n_net = yaml_obj["n_net"]
top_n = yaml_obj["top_n"]

LabelType = yaml_obj["LabelType"]

# max or mean, min  MeanForEachFutureFate ,MeanForAllCells
anal_type = yaml_obj["anal_type"]
Normalize = yaml_obj["Normalize"]  # RMS, Raw

#sum_mean_switch = yaml_obj["sum_mean_switch"]
ext_switch = yaml_obj["ext_switch"]

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
# Load parameterws from the first sample
base_dir = sample_list[0]+"/"
parameter_path = base_dir + "parameters.pickle"
param = sutil.PickleLoad(parameter_path)
hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature, reg, norm_final = gnn_models.LoadTrainingParameters(
    param)

num_time = len(time_list)

# +
# Define dir names
save_dir = "../bar_plot_result/" + save_dir_name + \
    "_sample=%d" % (sample_n) + "/%s/%s/%s/%s/" % (anal_type,
                                                   ModelType, LabelType, Normalize)
data_dir = save_dir + "data/"
fig_dir = save_dir + "fig/"

data_dir_bar_plot = save_dir + "data/bar_plot/"
sutil.MakeDirs(data_dir_bar_plot)

fig_dir_misc = save_dir + "fig/misc/"
sutil.MakeDirs(fig_dir_misc)


# +
# Cell external model or full model
# Reshape the IG data convenient for bar plot and save them.
for CorrectClass in [0, 1, 2]:

    y = sutil.PickleLoad(
        data_dir + "label=%d_average_mean.pickle" % CorrectClass)

    x = sutil.PickleLoad(
        data_dir + "label=%d_featurename_list.pickle" % CorrectClass)

    yerr = sutil.PickleLoad(
        data_dir + "label=%d_average_se.pickle" % CorrectClass)

    x_target_tmp = x
    y_target_tmp = y
    yerr_target_tmp = yerr

    # Extract only nonan values. Remove unncessary features with "Others" by this.
    n_nanind_list = np.where(~np.isnan(y_target_tmp))[0]
    # print(n_nanind_list)
    # for idx in n_nanind_list:
    # print(x_target_tmp[idx])

    # List of IGs of all the features
    x_target = []
    y_target = []
    yerr_target = []

    for n_nanind in n_nanind_list:
        label = x_target_tmp[n_nanind]
        x_target.append(x_target_tmp[n_nanind])
        y_target.append(y_target_tmp[n_nanind])
        yerr_target.append(yerr_target_tmp[n_nanind])

    # Extract IGs for Random feature
    random_feature_index = []
    if ext_switch == "ext":
        for n_nanind in n_nanind_list:
            label = x_target_tmp[n_nanind]
            if ("Random" in label) == True:
                if ("Target cell" in label) == False:  # external model
                    random_feature_index.append(n_nanind)
    if ext_switch == "full":
        for n_nanind in n_nanind_list:
            label = x_target_tmp[n_nanind]
            if ("Random" in label) == True:
                random_feature_index.append(n_nanind)

    size_data = len(random_feature_index)
    log = pd.DataFrame([], columns=['Feature category',
                       'Mean of IG', 'SE of IG'], index=range(size_data))

    # print(log)
    for i, idx in enumerate(random_feature_index):
        addRow = [x_target_tmp[idx], y_target_tmp[idx], yerr_target_tmp[idx]]
        # print(addRow)
        log.iloc[i, :] = addRow

    filename_list_IG_random_feature = "list_sample_averaged_IG_random_features_label=%s.txt" % celltype_label[
        CorrectClass]
    log.to_csv(data_dir_bar_plot +
               filename_list_IG_random_feature, index=False)

    standard_array_max = np.array(
        y_target_tmp[random_feature_index]) + np.array(yerr_target_tmp[random_feature_index])
    standard_max = np.max(standard_array_max)

    standard_array_min = np.array(
        y_target_tmp[random_feature_index]) - np.array(yerr_target_tmp[random_feature_index])
    standard_min = np.min(standard_array_min)

    filename_standard = "standard_label=%s.txt" % celltype_label[CorrectClass]
    np.savetxt(data_dir_bar_plot + filename_standard,
               np.array([standard_min, standard_max]).reshape(1, 2), header="min,max")

    # Sort features without Zero feature and extract features relevant in exteranal model

    data_list_x = []
    data_list_y = []
    data_list_yerr = []

    data_list_feature_type = []

    for l in range(input_size):
        for k in [1, 2]:  # celltype_name_list[k]=["Others","Target cell","Neighbor cell"] Run only of relavant cell types

            x_target_sort = []
            y_target_sort = []
            yerr_target_sort = []

            feature_type_target_sort = []

            for j in range(num_time):

                time = j-num_time + 1
                feature_type = "%s of %s at t=%d" % (
                    feature_list[l], celltype_name_list[k], time)

                if (feature_list[l] != "Zero") and (feature_list[l] != "Random"):

                    # Remove Del Target
                    if (feature_list[l] != "Del") or (celltype_name_list[k] != "Target cell"):

                        if (feature_list[l] != "NB") or (time != 0):

                            if (feature_list[l] != "Del") or (time != 0):

                                if (feature_list[l] != "Div") or (time != 0):

                                    if ext_switch == "ext":

                                        # For ext model
                                        if celltype_name_list[k] != "Target cell":

                                            idx = x_target.index(feature_type)

                                            x_target_sort.append(x_target[idx])
                                            y_target_sort.append(y_target[idx])
                                            yerr_target_sort.append(
                                                yerr_target[idx])

                                            feature_type_target_sort.append(
                                                feature_type)
                                    elif ext_switch == "full":  # For full model
                                        idx = x_target.index(feature_type)

                                        x_target_sort.append(x_target[idx])
                                        y_target_sort.append(y_target[idx])
                                        yerr_target_sort.append(
                                            yerr_target[idx])

                                        feature_type_target_sort.append(
                                            feature_type)

            if len(x_target_sort) > 0:

                data_list_x.append(x_target_sort)
                data_list_y.append(y_target_sort)
                data_list_yerr.append(yerr_target_sort)
                data_list_feature_type.append(feature_type_target_sort)

    # Save data

    filename_data_list_feature_type = "data_list_feature_type_label=%s.pickle" % celltype_label[
        CorrectClass]
    sutil.PickleDump(data_list_feature_type,
                     data_dir_bar_plot + filename_data_list_feature_type)
    filename_data_list_x = "data_list_x_label=%s.pickle" % celltype_label[CorrectClass]
    sutil.PickleDump(data_list_x, data_dir_bar_plot + filename_data_list_x)
    filename_data_list_y = "data_list_y_label=%s.pickle" % celltype_label[CorrectClass]
    sutil.PickleDump(data_list_y, data_dir_bar_plot + filename_data_list_y)
    filename_data_list_yerr = "data_list_yerr_label=%s.pickle" % celltype_label[
        CorrectClass]
    sutil.PickleDump(data_list_yerr, data_dir_bar_plot +
                     filename_data_list_yerr)

    # Save data as table
    data_list_x_1D = [x for row in data_list_x for x in row]
    data_list_y_1D = [x for row in data_list_y for x in row]
    data_list_yerr_1D = [x for row in data_list_yerr for x in row]
    data_list_feature_type_1D = [
        x for row in data_list_feature_type for x in row]

    size_data = len(data_list_x_1D)
    log2 = pd.DataFrame([], columns=['Feature category',
                        'Mean of IG', 'SE of IG'], index=range(size_data))

    # print(log2)
    for i in range(size_data):

        addRow = [data_list_feature_type_1D[i],
                  data_list_y_1D[i], data_list_yerr_1D[i]]
        # print(addRow)
        log2.iloc[i, :] = addRow

    filename_IGs_ext_all = "IGs_label=%s_pre.txt" % celltype_label[CorrectClass]
    log2.to_csv(data_dir_bar_plot + filename_IGs_ext_all, index=False)

    # Resort the IGs array to make the order the same as the plot
    log3 = pd.DataFrame([], columns=['Feature category',
                        'Mean of IG', 'SE of IG'], index=range(size_data))

    count = 0
    for i, data_list_name in enumerate(data_list_x):

        y1 = data_list_y[i]
        yerr1 = data_list_yerr[i]
        feature_type1 = data_list_feature_type[i]
        if len(data_list_name) == num_time:
            for j in range(num_time):  # Inversed order
                time = -j
                y = y1[num_time-j-1]
                yerr = yerr1[num_time-j-1]
                feature_type_tmp = feature_type1[num_time-j-1]

                addRow = [feature_type_tmp, y, yerr]
                # print(addRow)
                log3.iloc[count, :] = addRow
                count += 1

        if len(data_list_name) == (num_time-1):
            for j in range(num_time-1):  # Inversed order
                time = -j-1
                y = y1[num_time-j-2]
                yerr = yerr1[num_time-j-2]
                feature_type_tmp = feature_type1[num_time-j-2]

                addRow = [feature_type_tmp, y, yerr]
                # print(addRow)
                log3.iloc[count, :] = addRow
                count += 1

    filename_IGs_ext_all_sort = "IGs_label=%s.txt" % celltype_label[CorrectClass]
    log3.to_csv(data_dir_bar_plot + filename_IGs_ext_all_sort, index=False)


# +
# Cell external model
# We used the figure genretated in this cell in the paper.
# filename example: feature_ZZFR_SampleAverage_IG_all_sample=6_standadized_woxlabel_group_nolegend_ylim_-0.20_0.20.png
if ext_switch == "ext":
    width_bar = 0.8/num_time

    # print(base_dir)

    title_fig = save_dir_name + "_SampleAverage_IG_all"

    fig2 = plt.figure(figsize=(figx*figx_factor, figy*figy_factor_wolabel))
    fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig2.subplots_adjust(hspace=0.1, wspace=0.3)

    for CorrectClass in [0, 1, 2]:

        # Load data
        filename_data_list_feature_type = "data_list_feature_type_label=%s.pickle" % celltype_label[
            CorrectClass]
        data_list_feature_type = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_feature_type)
        filename_data_list_x = "data_list_x_label=%s.pickle" % celltype_label[CorrectClass]
        data_list_x = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_x)
        filename_data_list_y = "data_list_y_label=%s.pickle" % celltype_label[CorrectClass]
        data_list_y = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_y)
        filename_data_list_yerr = "data_list_yerr_label=%s.pickle" % celltype_label[
            CorrectClass]
        data_list_yerr = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_yerr)

        filename_standard = "standard_label=%s.txt" % celltype_label[CorrectClass]
        standard_min, standard_max = np.loadtxt(
            data_dir_bar_plot + filename_standard)

        num_data_select = len(data_list_x)

        ax2 = fig2.add_subplot(3, 1, 3-CorrectClass)

        ax1 = ax2.twinx()

        ax1.set_ylabel("%s" %
                       celltype_list[CorrectClass], fontsize=font_normal*0.5)
        ax1.yaxis.set_label_position("left")

        ax1.tick_params(left=False, right=False,
                        labelleft=False, labelright=False)

        ax2.grid(axis='y', c='gray', lw=0.25)
        ax2.set_axisbelow(True)

        for i, data_list_name in enumerate(data_list_x):

            # print(data_list_name)

            y1 = data_list_y[i]
            yerr1 = data_list_yerr[i]
            if len(data_list_name) == num_time:
                for j in range(num_time):  # Inversed order
                    time = -j
                    y = y1[num_time-j-1]
                    yerr = yerr1[num_time-j-1]
                    ax2.bar(i-(num_time-1)*width_bar/2+j*width_bar, y, yerr=yerr, width=width_bar, label="t=%d" % time,
                            capsize=capsize, color=color_list[j], ecolor="k", error_kw=dict(lw=0.5, capsize=1, capthick=0.5))

            if len(data_list_name) == (num_time-1):
                for j in range(num_time-1):  # Inversed order
                    time = -j-1
                    y = y1[num_time-j-2]
                    yerr = yerr1[num_time-j-2]
                    ax2.bar(i-(num_time-2)*width_bar/2+j*width_bar, y, yerr=yerr, width=width_bar, label="t=%d" % time,
                            capsize=capsize, color=color_list[j+1], ecolor="k", error_kw=dict(lw=0.5, capsize=1, capthick=0.5))

        for x_index in range(num_data_select):
            ax2.axvline(x=x_index+0.5, lw=0.25, c="gray", linestyle="--")

        ax2.tick_params(pad=pad, left=False, right=True,
                        labelleft=False, labelright=True)

        ax2.yaxis.set_label_position("right")

        plt.setp(ax2.get_yticklabels(), rotation=90, va='center')

        ax2.axhline(y=0, lw=axhline_lw, c="k")

        if Normalize == "Raw":
            ax2.axhspan(standard_min, standard_max,
                        color="orange", alpha=0.7, linewidth=0)

        if Normalize == "RMS":
            ax2.axhspan(-1, 1, color="orange", alpha=0.7, linewidth=0)

        # print(num_data_select)
        ax2.set_xlim([-0.5, num_data_select-0.5])

        ax2.set_ylim([ymin_set, ymax_set])
        yticks = ax2.get_yticks()
        # set new tick positions
        ax2.set_yticks(yticks[::len(yticks) // nbin_y])

        if CorrectClass == 1:
            ax2.set_ylabel("Integrated gradient", fontsize=font_normal)

        ax2.tick_params(labelbottom=False, bottom=False)

    filename_fig = title_fig + \
        "_sample=%d_standadized_woxlabel_group_nolegend_ylim_%.2f_%.2f" % (
            sample_n, ymin_set, ymax_set) + ".png"
    plt.savefig(fig_dir + filename_fig, format="png",
                dpi=dpi, transparent=True)
    filename_fig = title_fig + \
        "_sample=%d_standadized_woxlabel_group_nolegend_ylim_%.2f_%.2f" % (
            sample_n, ymin_set, ymax_set) + ".pdf"
    plt.savefig(fig_dir + filename_fig, format="pdf", transparent=True)

    plt.show()
    plt.close()


# +
# Cell external model
# filename example: feature_ZZFR_SampleAverage_IG_all_sample=6_standadized_group_nolegend_ylim_-0.20_0.20.png
# filename example: feature_ZZFR_SampleAverage_IG_all_sample=6_standadized_group_nolegend_ylim_-0.20_0.20_rotate.png

if ext_switch == "ext":
    width_bar = 0.8/num_time

    # print(base_dir)

    title_fig = save_dir_name + "_SampleAverage_IG_all"
    # print(cellname_list2_sort)
    fig2 = plt.figure(figsize=(figx*figx_factor, figy*figy_factor_wolabel))
    fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig2.subplots_adjust(hspace=0.1, wspace=0.3)

    for CorrectClass in [0, 1, 2]:

        # Load data
        filename_data_list_feature_type = "data_list_feature_type_label=%s.pickle" % celltype_label[
            CorrectClass]
        data_list_feature_type = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_feature_type)
        filename_data_list_x = "data_list_x_label=%s.pickle" % celltype_label[CorrectClass]
        data_list_x = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_x)
        filename_data_list_y = "data_list_y_label=%s.pickle" % celltype_label[CorrectClass]
        data_list_y = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_y)
        filename_data_list_yerr = "data_list_yerr_label=%s.pickle" % celltype_label[
            CorrectClass]
        data_list_yerr = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_yerr)

        filename_standard = "standard_label=%s.txt" % celltype_label[CorrectClass]
        standard_min, standard_max = np.loadtxt(
            data_dir_bar_plot + filename_standard)

        num_data_select = len(data_list_x)

        ax2 = fig2.add_subplot(3, 1, 3-CorrectClass)

        ax1 = ax2.twinx()

        ax1.set_ylabel("%s" %
                       celltype_list[CorrectClass], fontsize=font_normal*0.5)
        ax1.yaxis.set_label_position("left")

        ax1.tick_params(left=False, right=False,
                        labelleft=False, labelright=False)

        ax2.grid(axis='y', c='gray', lw=0.25)
        ax2.set_axisbelow(True)
        #ax2.errorbar(x_target,y_target,yerr=yerr_target,capsize=5, fmt='o', markersize=10, ecolor='black',markeredgecolor = "black", color='w')
        # ax2.bar(x,y,yerr=yerr,capsize=capsize,color='#377eb8',ecolor="k")

        for i, data_list_name in enumerate(data_list_x):

            # print(data_list_name)

            y1 = data_list_y[i]
            yerr1 = data_list_yerr[i]
            if len(data_list_name) == num_time:
                for j in range(num_time):  # Inersed order
                    time = -j
                    y = y1[num_time-j-1]
                    yerr = yerr1[num_time-j-1]
                    ax2.bar(i-(num_time-1)*width_bar/2+j*width_bar, y, yerr=yerr, width=width_bar, label="t=%d" % time,
                            capsize=capsize, color=color_list[j], ecolor="k", error_kw=dict(lw=0.5, capsize=1, capthick=0.5))

                    #ax2.text(i-(num_time-1)*width_bar/2+j*width_bar, y, y, ha='center', va='bottom')
            if len(data_list_name) == (num_time-1):
                for j in range(num_time-1):  # Inersed order
                    time = -j-1
                    y = y1[num_time-j-2]
                    yerr = yerr1[num_time-j-2]
                    ax2.bar(i-(num_time-2)*width_bar/2+j*width_bar, y, yerr=yerr, width=width_bar, label="t=%d" % time,
                            capsize=capsize, color=color_list[j+1], ecolor="k", error_kw=dict(lw=0.5, capsize=1, capthick=0.5))

                    #ax2.text(i-(num_time-1)*width_bar/2+j*width_bar, y, y, ha='center', va='bottom')

        for x_index in range(num_data_select):
            ax2.axvline(x=x_index+0.5, lw=0.25, c="gray", linestyle="--")
        #ax2.set_xticklabels(x, rotation=90)

        # ax.set_ylim([-np.nanmax(IG_max_average_sort[top_index])*2.5,np.nanmax(IG_max_average_sort[top_index])*2.5])

        ax2.tick_params(pad=pad, left=False, right=True,
                        labelleft=False, labelright=True)

        ax2.yaxis.set_label_position("right")

        #plt.setp(ax2.get_xticklabels(), rotation=90, va='center')
        #plt.setp(ax2.get_yticklabels(), rotation=90, va='center', ha="center")
        plt.setp(ax2.get_yticklabels(), rotation=90, va='center')

        ax2.axhline(y=0, lw=axhline_lw, c="k")

        #ax2.axhspan(standard_min, standard_max, color = "red",alpha=0.5)
        if Normalize == "Raw":
            ax2.axhspan(standard_min, standard_max,
                        color="orange", alpha=0.7, linewidth=0)

        if Normalize == "RMS":
            ax2.axhspan(-1, 1, color="orange", alpha=0.7, linewidth=0)

        #ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.5)

        # ax2.axhline(y=standard_max,lw=0.25,c="r")
        # ax2.axhline(y=standard_min,lw=0.25,c="r")

        # print(num_data_select)
        ax2.set_xlim([-0.5, num_data_select-0.5])

        ax2.set_ylim([ymin_set, ymax_set])
        yticks = ax2.get_yticks()
        # set new tick positions
        ax2.set_yticks(yticks[::len(yticks) // nbin_y])

        # ax2.set_ylim([ymin_set,ymax_set])

        # ax2.set_ylabel("%s"%celltype_list[CorrectClass])

        #ax2.set_title("%d samples:"%sample_n+save_dir_name +"_" +celltype_list[CorrectClass])
        if CorrectClass == 1:
            ax2.set_ylabel("Integrated gradient", fontsize=font_normal)

        # if CorrectClass==0:
        #ax1.set_xticklabels(celltype_list, rotation=90)

        # ax1.set_xlabel("test")

        #xticks_position = [(i-(num_time-1)*width_bar/2+(num_time/2)*width_bar) for i in range(3)]

        if CorrectClass == 0:
            #ax2.set_xlabel("NB\n\nDel\n\nDiv", rotation=90)

            # plt.xticks([0,1,2],["NB","Del","Div"])
            # plt.xticks(rotation=90)
            ax2.xaxis.set_tick_params(rotation=90, labelsize=font_normal*0.8)
            plt.xticks([0, 1, 2], celltype_label)

        if CorrectClass != 0:
            ax2.tick_params(labelbottom=False, bottom=False)

    filename_fig = title_fig + \
        "_sample=%d_standadized_group_nolegend_ylim_%.2f_%.2f" % (
            sample_n, ymin_set, ymax_set) + ".png"
    plt.savefig(fig_dir + filename_fig, format="png",
                dpi=dpi, transparent=True)

    filename_fig = title_fig + \
        "_sample=%d_standadized_group_nolegend_ylim_%.2f_%.2f" % (
            sample_n, ymin_set, ymax_set) + ".pdf"
    plt.savefig(fig_dir + filename_fig, format="pdf", transparent=True)

    plt.show()
    plt.close()

    # Rotate figure
    from PIL import Image

    filename_fig = title_fig + \
        "_sample=%d_standadized_group_nolegend_ylim_%.2f_%.2f" % (
            sample_n, ymin_set, ymax_set) + ".png"
    im = Image.open(fig_dir + filename_fig)

    img = np.array(im)

    filename_fig = title_fig + \
        "_sample=%d_standadized_group_nolegend_ylim_%.2f_%.2f_rotate" % (
            sample_n, ymin_set, ymax_set) + ".png"
    Image.fromarray(np.rot90(img, 3)).save(fig_dir + filename_fig)


# +
# Full model (not external model)
# Added for sum model
if ext_switch == "full":

    # if sum_mean_switch == "sum":
    width_bar = 0.8/num_time

    # print(base_dir)

    title_fig = save_dir_name + "_SampleAverage_IG_all"
    # print(cellname_list2_sort)
    fig2 = plt.figure(figsize=(figx*figx_factor, figy*figy_factor_wolabel))
    fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig2.subplots_adjust(hspace=0.1, wspace=0.3)

    for CorrectClass in [0, 1, 2]:

        # Load data
        filename_data_list_feature_type = "data_list_feature_type_label=%s.pickle" % celltype_label[
            CorrectClass]
        data_list_feature_type = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_feature_type)
        filename_data_list_x = "data_list_x_label=%s.pickle" % celltype_label[CorrectClass]
        data_list_x = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_x)
        filename_data_list_y = "data_list_y_label=%s.pickle" % celltype_label[CorrectClass]
        data_list_y = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_y)
        filename_data_list_yerr = "data_list_yerr_label=%s.pickle" % celltype_label[
            CorrectClass]
        data_list_yerr = sutil.PickleLoad(
            data_dir_bar_plot + filename_data_list_yerr)
        filename_standard = "standard_label=%s.txt" % celltype_label[CorrectClass]
        standard_min, standard_max = np.loadtxt(
            data_dir_bar_plot + filename_standard)
        num_data_select = len(data_list_x)

        ax2 = fig2.add_subplot(3, 1, 3-CorrectClass)

        ax1 = ax2.twinx()

        ax1.set_ylabel("%s" %
                       celltype_list[CorrectClass], fontsize=font_normal)
        ax1.yaxis.set_label_position("left")

        ax1.tick_params(left=False, right=False,
                        labelleft=False, labelright=False)

        ax2.grid(axis='y', c='gray', lw=0.25)
        ax2.set_axisbelow(True)
        #ax2.errorbar(x_target,y_target,yerr=yerr_target,capsize=5, fmt='o', markersize=10, ecolor='black',markeredgecolor = "black", color='w')
        # ax2.bar(x,y,yerr=yerr,capsize=capsize,color='#377eb8',ecolor="k")

        for i, data_list_name in enumerate(data_list_x):

            # print(data_list_name)

            y1 = data_list_y[i]
            yerr1 = data_list_yerr[i]
            if len(data_list_name) == num_time:
                for j in range(num_time):  # Inersed order
                    time = -j
                    y = y1[num_time-j-1]
                    yerr = yerr1[num_time-j-1]
                    ax2.bar(i-(num_time-1)*width_bar/2+j*width_bar, y, yerr=yerr, width=width_bar, label="t=%d" % time,
                            capsize=capsize, color=color_list[j], ecolor="k", error_kw=dict(lw=0.5, capsize=1, capthick=0.5))

            if len(data_list_name) == (num_time-1):
                for j in range(num_time-1):  # Inersed order
                    time = -j-1
                    y = y1[num_time-j-2]
                    yerr = yerr1[num_time-j-2]
                    ax2.bar(i-(num_time-2)*width_bar/2+j*width_bar, y, yerr=yerr, width=width_bar, label="t=%d" % time,
                            capsize=capsize, color=color_list[j+1], ecolor="k", error_kw=dict(lw=0.5, capsize=1, capthick=0.5))

        for x_index in range(num_data_select):
            ax2.axvline(x=x_index+0.5, lw=0.25, c="gray", linestyle="--")
        #ax2.set_xticklabels(x, rotation=90)

        # ax.set_ylim([-np.nanmax(IG_max_average_sort[top_index])*2.5,np.nanmax(IG_max_average_sort[top_index])*2.5])

        ax2.tick_params(pad=pad, left=False, right=True,
                        labelleft=False, labelright=True)

        ax2.yaxis.set_label_position("right")

        #plt.setp(ax2.get_xticklabels(), rotation=90, va='center')
        #plt.setp(ax2.get_yticklabels(), rotation=90, va='center', ha="center")
        plt.setp(ax2.get_yticklabels(), rotation=90, va='center')

        ax2.axhline(y=0, lw=axhline_lw, c="k")

        #ax2.axhspan(standard_min, standard_max, color = "red",alpha=0.5)
        ax2.axhspan(standard_min, standard_max,
                    color="orange", alpha=0.7, linewidth=0)

        #ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.5)

        # ax2.axhline(y=standard_max,lw=0.25,c="r")
        # ax2.axhline(y=standard_min,lw=0.25,c="r")

        # print(num_data_select)
        ax2.set_xlim([-0.5, num_data_select-0.5])

        yticks = ax2.get_yticks()
        # ax2.set_yticks(yticks[::len(yticks) // nbin_y]) # set new tick positions

        ax2.set_ylim([ymin_set, ymax_set])

        # ax2.set_ylabel("%s"%celltype_list[CorrectClass])

        #ax2.set_title("%d samples:"%sample_n+save_dir_name +"_" +celltype_list[CorrectClass])
        if CorrectClass == 1:
            ax2.set_ylabel("Integrated gradient", fontsize=font_normal)

        # if CorrectClass==0:
            # ax2.set_xlabel("Features",rotation=180)

        # if CorrectClass!=0:
        ax2.tick_params(labelbottom=False, bottom=False)

    filename_fig = title_fig + \
        "_sample=%d_standadized_woxlabel_group_nolegend_noscale_full_model_%.2f_%.2f" % (
            sample_n, ymin_set, ymax_set) + ".png"
    plt.savefig(fig_dir + filename_fig, format="png",
                dpi=dpi, transparent=True)
    filename_fig = title_fig + \
        "_sample=%d_standadized_woxlabel_group_nolegend_noscale_full_model_%.2f_%.2f" % (
            sample_n, ymin_set, ymax_set) + ".pdf"
    plt.savefig(fig_dir + filename_fig, format="pdf", transparent=True)

    plt.show()
    plt.close()

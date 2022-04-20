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

""" Plot peformance averaged over samples"""

import matplotlib.pyplot as plt
import numpy as np
from functions import system_utility as sutil
import os
import sys
version = 1
print("version=%d" % version)

feature_compare = 0


# parameters for plot
filename_fig_yml = 'input_fig_config_plot_auc.yml'
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

ymin = yaml_fig_obj["ymin"]
ymax = yaml_fig_obj["ymax"]


# parameters for plot
filename_fig_yml = 'input_fig_config.yml'
path_fig_yml = "./" + filename_fig_yml
yaml_fig_obj = sutil.LoadYml(path_fig_yml)

xlabel = yaml_fig_obj["xlabel"]
ylabel = yaml_fig_obj["ylabel"]

name_list = yaml_fig_obj["name_list"]


plt.rcParams['font.family'] = yaml_fig_obj["font.family"]
plt.rcParams['xtick.direction'] = yaml_fig_obj["xtick.direction"]
plt.rcParams['ytick.direction'] = yaml_fig_obj["ytick.direction"]
plt.rcParams['xtick.major.width'] = yaml_fig_obj["xtick.major.width"]
plt.rcParams['ytick.major.width'] = yaml_fig_obj["ytick.major.width"]
plt.rcParams['font.size'] = yaml_fig_obj["font.size"]
plt.rcParams['axes.linewidth'] = yaml_fig_obj["axes.linewidth"]


# +
# parameters

filename_yml = 'input_analyze_prediction_performance.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

base_name = yaml_obj["base_name"]
dir_base_list = yaml_obj["dir_base_list"]

index_list = yaml_obj["index_list"]

work_dir = yaml_obj["work_dir"]

celltype_list = yaml_obj["celltype_list"]
dict_list = yaml_obj["dict_list"]


data_type = base_name
version = "Performance-%s-%.2f" % (data_type, axis_width)

listname_list = ["%s-sample%d" % (data_type, i) for i in index_list]


title_ave = "%d-Average:%s" % (len(listname_list), data_type)

n_sample = len(listname_list)
n_xlabel = len(name_list)

n_legend = len(dict_list)


# +
filename_AUCForMaxAUC = "/ROC_time/data/Max_AUC.pickle"
filename_acc_test = "/log/history_acc_balanced_eval.txt"
filename_acc_train = "/log/history_acc_balanced_train.txt"

filename_Max_AUC_epoch = "/ROC_time/data/Max_AUC_epoch.pickle"

filename_AUC_time = "/ROC_time/data/AUC_time.pickle"

filename_Max_ACC_epoch = "/ROC_time/data/Max_BalancedACC_epoch.pickle"  # Balanced ACC
filename_Max_ACC = "/ROC_time/data/Max_BalancedACC.pickle"  # Balanced ACC
filename_AUCForMaxACC = "/ROC_time/data/AUCForMaxBalancedACC.pickle"  # AUCForBalanced ACC


filename_MaxF1 = "/ROC_time/data/MaxMacroF1.pickle"
filename_AUCForMaxMacroF1 = "/ROC_time/data/AUCForMaxMacroF1.pickle"


filename_parameters = "/parameters.pickle"

all_data = np.zeros((n_sample, n_legend, n_xlabel))

cmap = ["#DDAA33", "#004488", "#BB5566"]  # Tol contrast
marker_type_list = ["o", "D", "s"]
msize_list = [msize, msize*0.9, msize*0.9]


# -

def PlotSingleValuePerformanceAveragedOverSamples(listname_list, data_type_name, version, work_dir, filename_Max_ACC, yaxis_name):
    """
    Plot single value peformance such as ACC, F1 score averaged over samples.
    """

    sample = 0

    for i, listname in enumerate(listname_list):

        save_dir = work_dir + \
            "/summary_%s/%s/%s/" % (version, data_type_name, listname)
        # sutil.MakeDirs(save_dir)

        path = work_dir + "/%s.txt" % listname
        dir_list = sutil.ReadLinesText(path)

        dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

        if i == 0:
            data_all_list = np.zeros((len(listname_list), len(dir_list)))

        for j, dir_path in enumerate(dir_list):
            data = sutil.PickleLoad(dir_path + filename_Max_ACC)

            data_all_list[i, j] = data

    # Average data_all_list
    # print(data_all_list)
    # print(np.mean(data_all_list,axis=0))
    mean_list = np.mean(data_all_list, axis=0)
    std_list = np.std(data_all_list, axis=0, ddof=1)

    # Plot Max of ACC (balanced ACC)
    # Average data_all_list and plot, save the values
    save_dir = work_dir + "/summary_%s/%s/all/" % (version, data_type_name)
    sutil.MakeDirs(save_dir)

    save_dir_figs = save_dir + "figs/"
    save_dir_data = save_dir + "data/"

    sutil.MakeDirs(save_dir_figs)
    sutil.MakeDirs(save_dir_data)

    np.savetxt(save_dir_data + "data_all_list.txt", data_all_list)

    np.savetxt(save_dir_data + "data_all_list_mean.txt", mean_list)
    np.savetxt(save_dir_data + "data_all_list_std.txt", std_list)

    fig = plt.figure(figsize=(figx*fig_factor, figy*fig_factor))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    ax.errorbar(name_list, mean_list, std_list,
                fmt="o", capsize=2, markersize=msize)

    ax.set_xticks(name_list)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(yaxis_name)
    ax.set_title(title_ave)
    ax.set_ylim(ymin=0.3, ymax=0.6)

    figname = save_dir_figs + "%s_%s_legend.pdf" % (yaxis_name, data_type_name)
    plt.savefig(figname, transparent=True)
    figname_png = save_dir_figs + \
        "%s_%s_legend.png" % (yaxis_name, data_type_name)
    plt.savefig(figname_png, dpi=dpi, transparent=True)

    plt.show()

    plt.close()

# +
# Plot Max of balanced ACC
# Load Max ACC from sample list to data_all_list


data_type_name = "MaxBalancedACC"
yaxis_name = "Balanced_ACC"
PlotSingleValuePerformanceAveragedOverSamples(
    listname_list, data_type_name, version, work_dir, filename_Max_ACC, yaxis_name)


# +
# Plot Max of F1 score
# Load Max F1 score from sample list to data_all_list
# Average data_all_list and plot, save the values

data_type_name = "MaxF1Score"
yaxis_name = "F1Score"
PlotSingleValuePerformanceAveragedOverSamples(
    listname_list, data_type_name, version, work_dir, filename_MaxF1, yaxis_name)


# +
def PlotAUCAveragedOverSamples(filename, listname_list, data_type_name, version, work_dir):
    """
    Plot AUC averaged over samples.
    """

    # Plot AUCs for Max AUC models for each sample
    sample = 0
    for listname in listname_list:

        save_dir = work_dir + \
            "/summary_%s/%s/%s/" % (version, data_type_name, listname)
        sutil.MakeDirs(save_dir)

        save_dir_figs = save_dir + "figs/"
        save_dir_data = save_dir + "data/"

        sutil.MakeDirs(save_dir_figs)
        sutil.MakeDirs(save_dir_data)

        path = work_dir + "/%s.txt" % listname
        #path = work_dir + "/SelfInfo_list.txt"
        dir_list = sutil.ReadLinesText(path)

        dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)
        fig = plt.figure(figsize=(figx*fig_factor, figy*fig_factor))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

        count_dict = 0
        for i in dict_list:

            auc_list = []
            features_list = []

            labels = []

            x_list = []
            count = 0
            for dir_path in dir_list:

                dir_path = dir_path + "/"
                celltype = celltype_list[count_dict]

                file_path = dir_path + filename
                auc_dict = sutil.PickleLoad(file_path)

                # print(auc_dict['macro'])
                # print(auc_dict)
                auc = auc_dict[i]

                auc_list.append(auc)

                x_list.append(count)

                if count_dict == 0:
                    label_name = "%d:" % count
                    ax.scatter(count, auc, label=label_name, color="none")

                    labels.append(label_name)

                count += 1

                # print(name_list)
                # print(labels)

            all_data[sample, count_dict, :] = auc_list

            if feature_compare == 0:
                ax.plot(name_list, auc_list, marker=marker_type_list[count_dict], label=celltype,
                        c=cmap[count_dict], markersize=msize_list[count_dict], linestyle='None')

            if feature_compare == 1:
                ax.plot(x_list, auc_list, marker=marker_type_list[count_dict], label=celltype,
                        c=cmap[count_dict], markersize=msize_list[count_dict], linestyle='None')

            x_filename = "%s_x.txt" % celltype
            y_filename = "%s_y.txt" % celltype

            np.savetxt(save_dir_data + x_filename, x_list)
            np.savetxt(save_dir_data + y_filename, auc_list)

            ax.set_xticks(x_list)
           # ax.set_xlim([0.0, 1.0])
            ax.set_ylim([ymin, ymax])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('Max AUC:%s' % listname)
            # print(labels)
            # if i == 0:
            #plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))

            plt.legend(celltype_list, loc="upper right",
                       fontsize=legend_size, framealpha=0)

            count_dict += 1

            # print(features_list)
        ax.axhline(0.5, ls="--", color="k", lw=0.75)

        figname = save_dir_figs + "%s.pdf" % data_type_name
        plt.savefig(figname, transparent=True)
        figname_png = save_dir_figs + "%s.png" % data_type_name
        plt.savefig(figname_png, dpi=dpi, transparent=True)

        plt.show()

        plt.close()

        ##### To show only legend ##########
        fig = plt.figure(figsize=(figx*fig_factor, figy*fig_factor))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

        count_dict = 0
        for i in dict_list:

            auc_list = []
            features_list = []

            labels = []

            x_list = []
            count = 0
            for dir_path in dir_list:

                dir_path = dir_path + "/"

                celltype = celltype_list[count_dict]

                features = sutil.ReadLinesText(
                    dir_path + "features/features.txt")
                # print(features)

                file_path = dir_path + filename
                auc_dict = sutil.PickleLoad(file_path)
                # print(auc_dict)
                auc = auc_dict[i]

                auc_list.append(auc)

                x_list.append(count)

                if i == 0:
                    label_name = "%d:" % count
                    ax.scatter(count, auc, label=label_name, color="none")

                    labels.append(label_name)

                count += 1

            #ax.plot(x_list,auc_list,marker = "o", label = celltype)

            x_filename = "%s_x.txt" % celltype
            y_filename = "%s_y.txt" % celltype

            np.savetxt(save_dir_data + x_filename, x_list)
            np.savetxt(save_dir_data + y_filename, auc_list)

            ax.set_xticks(x_list)
           # ax.set_xlim([0.0, 1.0])
            #ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Condition')
            ax.set_ylabel('AUC')
            ax.set_title('AUC:%s' % listname)
            # print(labels)
            if i == 0:
                plt.legend(labels, loc='upper left',
                           fontsize=legend_size, framealpha=0)

            #plt.legend(celltype_list,loc="lower right",fontsize=10)

            # print(features_list)

        figname = save_dir_figs + "%s_legend.pdf" % data_type_name
        plt.savefig(figname, transparent=True)
        figname_png = save_dir_figs + "%s_legend.png" % data_type_name
        plt.savefig(figname_png, dpi=dpi, transparent=True)

        plt.show()

        plt.close()

        sample += 1

    # Plot AUCs for Max AUC models averaged over samples

    save_dir = work_dir + \
        "/summary_%s/%s/all/" % (version, data_type_name)+data_type + "/"
    sutil.MakeDirs(save_dir)

    save_dir_figs = save_dir + "figs/"
    save_dir_data = save_dir + "data/"

    sutil.MakeDirs(save_dir_figs)
    sutil.MakeDirs(save_dir_data)

    np.save(save_dir_data + "all.npy", all_data)

    fig = plt.figure(figsize=(figx*fig_factor, figy*fig_factor))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    count_dict = 0
    for i in dict_list:

        celltype = celltype_list[count_dict]

        all_data_ave = np.average(all_data, axis=0)

        all_data_std = np.std(all_data, axis=0, ddof=1)

        if feature_compare == 0:
            ax.errorbar(name_list, all_data_ave[count_dict], yerr=all_data_std[count_dict], fmt=marker_type_list[count_dict],
                        label=celltype, c=cmap[count_dict], capsize=2, markersize=msize_list[count_dict])
        if feature_compare == 1:
            ax.errorbar(x_list, all_data_ave[count_dict], yerr=all_data_std[count_dict], fmt=marker_type_list[count_dict],
                        label=celltype, c=cmap[count_dict], capsize=2, markersize=msize_list[count_dict])

        x_filename = "%s_x.txt" % celltype
        y_filename = "%s_y.txt" % celltype

        yerr_filename = "%s_yerr.txt" % celltype

        np.savetxt(save_dir_data + x_filename, x_list)
        np.savetxt(save_dir_data + y_filename, all_data_ave[count_dict])
        np.savetxt(save_dir_data + yerr_filename, all_data_std[count_dict])

        ax.set_xticks(x_list)
       # ax.set_xlim([0.0, 1.0])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title_ave)

        count_dict = count_dict + 1

    ax.axhline(0.5, ls="--", color="k", lw=0.75)
    figname = save_dir_figs + "%s_no_legend.pdf" % data_type_name
    plt.savefig(figname, transparent=True)
    figname_png = save_dir_figs + "%s_no_legend.png" % data_type_name
    plt.savefig(figname_png, dpi=dpi, transparent=True)

    plt.legend(["random"] + celltype_list, loc="upper right",
               fontsize=legend_size, framealpha=0)

    figname = save_dir_figs + "%s.pdf" % data_type_name
    plt.savefig(figname, transparent=True)
    figname_png = save_dir_figs + "%s.png" % data_type_name
    plt.savefig(figname_png, dpi=dpi, transparent=True)

    plt.show()

    plt.close()


# -
filename = filename_AUCForMaxAUC
data_type_name = "AUCForMaxAUCModel"
PlotAUCAveragedOverSamples(filename, listname_list,
                           data_type_name, version, work_dir)

# Filenema for MaxACC
filename = filename_AUCForMaxACC
data_type_name = "AUCForMaxACCModel"
PlotAUCAveragedOverSamples(filename, listname_list,
                           data_type_name, version, work_dir)

# Filenema for MaxF1
filename = filename_AUCForMaxMacroF1
data_type_name = "AUCForMaxMacroF1Model"
PlotAUCAveragedOverSamples(filename, listname_list,
                           data_type_name, version, work_dir)

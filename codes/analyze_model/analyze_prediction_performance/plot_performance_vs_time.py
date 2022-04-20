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

""" Plot model peformance vs time """

import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from training import gnn_models
from functions import system_utility as sutil
import os
import sys
version = 1
print("version=%d" % version)


# parameters for plot
filename_fig_yml = 'input_fig_config.yml'
path_fig_yml = "./" + filename_fig_yml
yaml_fig_obj = sutil.LoadYml(path_fig_yml)


fig_hspace = yaml_fig_obj["fig_hspace"]
fig_wspace = yaml_fig_obj["fig_wspace"]


figx_time = yaml_fig_obj["figx_time"]
figy_time = yaml_fig_obj["figy_time"]
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


xlabel = yaml_fig_obj["xlabel"]
ylabel = yaml_fig_obj["ylabel"]

name_list = yaml_fig_obj["name_list"]

weight_plot = yaml_fig_obj["weight_plot"]
balancedACCWeighted_plot = yaml_fig_obj["balancedACCWeighted_plot"]
NumberOfCorrectPrediction_plot = yaml_fig_obj["NumberOfCorrectPrediction_plot"]


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

gpu = yaml_obj["gpu"]

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
# Filenames for loading

#filename = "/ROC_time/data/Max_AUC.pickle"
filename_acc_test = "/log/history_acc_balanced_eval.txt"
filename_acc_train = "/log/history_acc_balanced_train.txt"

filename_Max_AUC_epoch = "/ROC_time/data/Max_AUC_epoch.pickle"
filename_AUC_time = "/ROC_time/data/AUC_time.pickle"

# Filenames for saving
filename_acc_test_txt = "BalancedAcc_test.txt"
filename_acc_train_txt = "BalancedAcc_train.txt"

filename_Max_AUC_epoch_txt = "Max_AUC_epoch.txt"
filename_AUC_time_txt = "AUC_time.txt"


filename_Max_ACC_epoch = "/ROC_time/data/Max_BalancedACC_epoch.pickle"  # Balanced ACC
filename_Max_ACC = "/ROC_time/data/Max_BalancedACC.pickle"  # Balanced ACC
filename_AUCForMaxACC = "/ROC_time/data/AUCForMaxBalancedACC.pickle"  # AUCForBalanced ACC

filename_Max_ACC_epoch_txt = "Max_ACC_epoch.txt"  # Balanced ACC
filename_Max_ACC_txt = "Max_ACC.txt"  # Balanced ACC
filename_AUCForMaxACC_txt = "AUCForMaxACC.txt"  # AUCForBalanced ACC


filename_MaxMacroF1_epoch = "/ROC_time/data/MaxMacroF1_epoch.pickle"
filename_MaxMacroF1 = "/ROC_time/data/MaxMacroF1.pickle"
filename_AUCForMaxMacroF1 = "/ROC_time/data/AUCForMaxMacroF1.pickle"

filename_MaxMacroF1_epoch_txt = "MaxMacroF1_epoch.txt"
filename_MaxMacroF1_txt = "MaxMacroF1.txt"
filename_AUCForMaxMacroF1_txt = "AUCForMaxMacroF1.txt"


filename_parameters = "/parameters.pickle"

all_data = np.zeros((n_sample, n_legend, n_xlabel))

cmap = ["#DDAA33", "#004488", "#BB5566"]  # Tol contrast
marker_type_list = ["o", "D", "s"]
msize_list = [msize, msize*0.9, msize*0.9]
# +


def LoadTrainingParameters(param):

    # Load parameters

    hid_node = param["hid_node"]

    time_list = param["time_list"]
    p_hidden = param["p_hidden"]
    in_plane = param["in_plane"]
    NoSelfInfo = param["NoSelfInfo"]
    n_layers = param["n_layers"]
    skip = param["skip"]
    input_size = param["input_size"]

    epoch_total = param["epoch_total"]
    network_name = param["network_num"]
    model_save_rate = param["model_save_rate"]
    average_switch = param["average_switch"]
    architecture = param["architecture"]
    # print(average_switch)
    feature_list_edge = param["feature_list_edge"]
    #print(feature_list_edge )
    feature_edge_concat = param["feature_edge_concat"]

    input_size_edge = param["input_size_edge"]

    edge_switch = param["edge_switch"]

    crop_width_list_train = param["crop_width_list_train"]
    crop_height_list_train = param["crop_height_list_train"]
    crop_width_list_test = param["crop_width_list_test"]
    crop_height_list_test = param["crop_height_list_test"]

    cuda_clear = param["cuda_clear"]
    reg = param["reg"]
    norm_final = param["norm_final"]

    in_feats = input_size
    feature_self = "AllZero"
    feature = 'feature_concat'

    return hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature


# +
# Check GPUs
if gpu != -1:  # if gpu==-1, use cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu

print(th.cuda.device_count(), "GPUs available")
print(th.__version__)  # 0.4.0

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)
print(th.cuda.current_device())

# +
# Plot the time evolution of recall curves for each label and blanced ACC curves.
# Load balanced ACC history and epoch for Max AUC from the results of the training.
# Save maximum balanced ACC and epoch for the max value.
# the data points are saved as "/labels/test/test_summary/test_NB_rate_list.pickle".


fig_title = "Recall_BalancedACC_time_evolution"
num_row = len(listname_list)
num_column = len(name_list)

# figx_time=3
# figy_time=3
fig = plt.figure(figsize=(figx_time*num_column, figy_time*num_row))
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.subplots_adjust(hspace=fig_hspace, wspace=fig_wspace)


save_dir = work_dir + "/summary_%s/%s/" % (version, fig_title)
sutil.MakeDirs(save_dir)

save_dir_figs = save_dir + "figs/"
save_dir_data = save_dir + "data/"

sutil.MakeDirs(save_dir_figs)
sutil.MakeDirs(save_dir_data)

count = 1
for i, listname in enumerate(listname_list):

    path = work_dir + "/%s.txt" % listname
    #path = "./SelfInfo_list.txt"
    dir_list = sutil.ReadLinesText(path)

    dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

    for j, dir_path in enumerate(dir_list):

        name = name_list[j]

        save_dir_data_sample = save_dir_data + name + "/sample%d/" % i
        sutil.MakeDirs(save_dir_data_sample)

        dir_path = dir_path + "/"

        parameters_file_path = dir_path + filename_parameters
        param = sutil.PickleLoad(parameters_file_path)

        model_save_rate = param["model_save_rate"]
        epoch_total = param["epoch_total"]
        index_log = list(range(0, epoch_total, model_save_rate))
        index_log.append(epoch_total-1)

        acc_test_file_path = dir_path + filename_acc_test

        if os.path.exists(acc_test_file_path) == True:
            acc_test_history = np.loadtxt(acc_test_file_path)  # balanced ACC
            acc_test_history_target = acc_test_history[index_log]

            max_acc = np.max(acc_test_history_target)
            argmax_acc = np.argmax(acc_test_history_target)

            epoch_maxacc = index_log[argmax_acc]

            print("Max BalancedACC%f at epoch %d" % (max_acc, epoch_maxacc))

            sutil.PickleDump(epoch_maxacc,  dir_path + filename_Max_ACC_epoch)
            sutil.PickleDump(max_acc,  dir_path + filename_Max_ACC)

            sutil.SaveText(epoch_maxacc,  save_dir_data_sample +
                           filename_Max_ACC_epoch_txt)
            #sutil.SaveText(max_acc,  save_dir_data_sample + filename_Max_ACC_txt)

            AUC_time_data = sutil.PickleLoad(dir_path + filename_AUC_time)

            AUCForMaxACC = [AUC_time_data[argmax_acc][0],
                            AUC_time_data[argmax_acc][1], AUC_time_data[argmax_acc][2]]
            print(AUCForMaxACC)
            sutil.PickleDump(AUCForMaxACC, dir_path + filename_AUCForMaxACC)

            Max_AUC_epoch = sutil.PickleLoad(dir_path + filename_Max_AUC_epoch)

            sutil.SaveText(Max_AUC_epoch, save_dir_data_sample +
                           filename_Max_AUC_epoch_txt)
            filename_Max_AUC_epoch_txt = "Max_AUC_epoch.txt"

            filename_labels_test_pre = "/labels/test/test_summary/CorrectVsPredict/test_CorrectVsPredictAll_epoch="
            filename_labels_train_pre = "/labels/training/training_summary/CorrectVsPredict/training_CorrectVsPredictAll_epoch="

            NB_rate_list = []
            Dif_rate_list = []
            Div_rate_list = []
            for index in index_log:
                filename_labels_test = filename_labels_test_pre + "%d.txt" % index
                confusion = np.loadtxt(dir_path + filename_labels_test)
                NB_rate = confusion[0, 0]/np.sum(confusion[0, :])
                Dif_rate = confusion[1, 1]/np.sum(confusion[1, :])
                Div_rate = confusion[2, 2]/np.sum(confusion[2, :])

                NB_rate_list.append(NB_rate)
                Dif_rate_list.append(Dif_rate)
                Div_rate_list.append(Div_rate)

            NB_rate_list_filename = "/labels/test/test_summary/test_NB_rate_list.pickle"
            Dif_rate_list_filename = "/labels/test/test_summary/test_Dif_rate_list.pickle"
            Div_rate_list_filename = "/labels/test/test_summary/test_Div_rate_list.pickle"

            NB_rate_list_filename_txt = "test_NB_recall.txt"
            Dif_rate_list_filename_txt = "test_Del_recall.txt"
            Div_rate_list_filename_txt = "test_Div_recall.txt"

            sutil.PickleDump(NB_rate_list, dir_path + NB_rate_list_filename)
            sutil.PickleDump(Dif_rate_list, dir_path + Dif_rate_list_filename)
            sutil.PickleDump(Div_rate_list, dir_path + Div_rate_list_filename)

            sutil.SaveListText(
                NB_rate_list, save_dir_data_sample + NB_rate_list_filename_txt)
            sutil.SaveListText(
                Dif_rate_list, save_dir_data_sample + Dif_rate_list_filename_txt)
            sutil.SaveListText(
                Div_rate_list, save_dir_data_sample + Div_rate_list_filename_txt)

            NB_rate_list_train = []
            Dif_rate_list_train = []
            Div_rate_list_train = []
            for index in index_log:
                filename_labels_train = filename_labels_train_pre + "%d.txt" % index
                confusion = np.loadtxt(dir_path + filename_labels_train)
                NB_rate_train = confusion[0, 0]/np.sum(confusion[0, :])
                Dif_rate_train = confusion[1, 1]/np.sum(confusion[1, :])
                Div_rate_train = confusion[2, 2]/np.sum(confusion[2, :])

                NB_rate_list_train.append(NB_rate_train)
                Dif_rate_list_train.append(Dif_rate_train)
                Div_rate_list_train.append(Div_rate_train)

            NB_rate_list_filename_train = "/labels/training/training_summary/train_NB_rate_list.pickle"
            Dif_rate_list_filename_train = "/labels/training/training_summary/train_Dif_rate_list.pickle"
            Div_rate_list_filename_train = "/labels/training/training_summary/train_Div_rate_list.pickle"

            NB_rate_list_filename_train_txt = "train_NB_recall.txt"
            Dif_rate_list_filename_train_txt = "train_Del_recall.txt"
            Div_rate_list_filename_train_txt = "train_Div_recall.txt"

            sutil.PickleDump(NB_rate_list_train, dir_path +
                             NB_rate_list_filename_train)
            sutil.PickleDump(Dif_rate_list_train, dir_path +
                             Dif_rate_list_filename_train)
            sutil.PickleDump(Div_rate_list_train, dir_path +
                             Div_rate_list_filename_train)

            sutil.SaveListText(
                NB_rate_list_train, save_dir_data_sample + NB_rate_list_filename_train_txt)
            sutil.SaveListText(
                Dif_rate_list_train, save_dir_data_sample + Dif_rate_list_filename_train_txt)
            sutil.SaveListText(
                Div_rate_list_train, save_dir_data_sample + Div_rate_list_filename_train_txt)

            #acc_test = np.loadtxt(dir_path + filename_acc_test)
            acc_train = np.loadtxt(dir_path + filename_acc_train)
            # x=range(len(acc_test))

            epoch_list_filename_txt = "epoch_list.txt"
            epoch_balancedACC_test_filename_txt = "test_BalancedACC.txt"
            epoch_balancedACC_train_filename_txt = "train_BalancedACC.txt"

            sutil.SaveListText(
                index_log, save_dir_data_sample + epoch_list_filename_txt)
            sutil.SaveListText(
                acc_train[index_log], save_dir_data_sample + epoch_balancedACC_train_filename_txt)
            sutil.SaveListText(
                acc_test_history_target, save_dir_data_sample + epoch_balancedACC_test_filename_txt)

            ax = fig.add_subplot(num_row, num_column, count)

            # Data points are plot at the rate of saving models.
            ax.plot(index_log, acc_train[index_log],
                    c="cyan", label="TrainBalancedACC")
            ax.plot(index_log, acc_test_history_target,
                    c="green", label="TestBalancedACC")
            # ax.plot(x,acc_train)

            ax.axvline(x=epoch_maxacc, ls="--", lw=0.5,
                       c="k", label="MaxTestBalancedACC")
            ax.axvline(x=Max_AUC_epoch, ls="-", lw=0.5, c="r", label="MaxAUC")

            ax.plot(index_log, NB_rate_list,
                    c=cmap[0], lw=1, label="TestNBRecall")
            ax.plot(index_log, Dif_rate_list,
                    c=cmap[1], lw=1, label="TestDifRecall")
            ax.plot(index_log, Div_rate_list,
                    c=cmap[2], lw=1, label="TestDivRecall")

            ax.plot(index_log, NB_rate_list_train,
                    c=cmap[0], lw=1, ls="--", label="TrainNBRecall")
            ax.plot(index_log, Dif_rate_list_train,
                    c=cmap[1], lw=1, ls="--", label="TrainDifRecall")
            ax.plot(index_log, Div_rate_list_train,
                    c=cmap[2], lw=1, ls="--", label="TrainDivRecall")

            # ax.set_xticks(x_list)
           # ax.set_xlim([0.0, 1.0])
            #ax.set_ylim([ymin, ymax])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("ACC")
            ax.set_title('%s\n%s' % (name, listname))
            # print(labels)
            # if i == 0:
            #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.legend(fontsize=5, loc='upper right')

            #plt.legend(celltype_list,loc="upper right",fontsize=legend_size,framealpha=0)

        count += 1

        # print(features_list)
    #ax.axhline(0.5, ls = "--", color = "k",lw=0.75)

figname = save_dir_figs + "%s.pdf" % fig_title
plt.savefig(figname, transparent=True)
figname_png = save_dir_figs + "%s.png" % fig_title
plt.savefig(figname_png, dpi=dpi, transparent=True)

plt.show()

plt.close()


# +
# Plot the time evolution of recall for each label and macro-F1 score.
# When precision is nan, we define F1 = 0.
# Save maximum macro-F1 score and epoch for the max value.
# the data points are saved as "/labels/test/test_summary/test_MacroF1_list.pickle" and "test_NB_precision_list.pickle".

fig_title = "Recall_MacroF1Score_time_evolution"

num_row = len(listname_list)
num_column = len(name_list)

# figx_time=3
# figy_time=3
fig = plt.figure(figsize=(figx_time*num_column, figy_time*num_row))

fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.subplots_adjust(hspace=fig_hspace, wspace=fig_wspace)


save_dir = work_dir + "/summary_%s/%s/" % (version, fig_title)
sutil.MakeDirs(save_dir)

save_dir_figs = save_dir + "figs/"
save_dir_data = save_dir + "data/"

sutil.MakeDirs(save_dir_figs)
sutil.MakeDirs(save_dir_data)

count = 1
for i, listname in enumerate(listname_list):

    path = work_dir + "/%s.txt" % listname
    #path = "./SelfInfo_list.txt"
    dir_list = sutil.ReadLinesText(path)

    dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

    for j, dir_path in enumerate(dir_list):

        name = name_list[j]
        save_dir_data_sample = save_dir_data + name + "/sample%d/" % i
        sutil.MakeDirs(save_dir_data_sample)

        dir_path = dir_path + "/"

        parameters_file_path = dir_path + filename_parameters
        param = sutil.PickleLoad(parameters_file_path)

        model_save_rate = param["model_save_rate"]
        epoch_total = param["epoch_total"]
        index_log = list(range(0, epoch_total, model_save_rate))
        index_log.append(epoch_total-1)

        epoch_list_filename_txt = "epoch_list.txt"
        sutil.SaveListText(index_log, save_dir_data_sample +
                           epoch_list_filename_txt)

        acc_test_file_path = dir_path + filename_acc_test

        if os.path.exists(acc_test_file_path) == True:

            filename_labels_test_pre = "/labels/test/test_summary/CorrectVsPredict/test_CorrectVsPredictAll_epoch="
            filename_labels_train_pre = "/labels/training/training_summary/CorrectVsPredict/training_CorrectVsPredictAll_epoch="

            # Calculate recall for test and training data for each label

            # For test data
            dir_base_train_test = "test/test_summary/test"

            NB_prec_list = []
            Dif_prec_list = []
            Div_prec_list = []

            NB_rec_list = []
            Dif_rec_list = []
            Div_rec_list = []

            MacroF1_list = []

            for index in index_log:
                filename_labels_test = filename_labels_test_pre + "%d.txt" % index
                confusion = np.loadtxt(dir_path + filename_labels_test)

                if np.sum(confusion[:, 0]) != 0:
                    NB_prec = confusion[0, 0]/np.sum(confusion[:, 0])
                else:
                    NB_prec = 0

                if np.sum(confusion[:, 1]) != 0:
                    Dif_prec = confusion[1, 1]/np.sum(confusion[:, 1])
                else:
                    Dif_prec = 0

                if np.sum(confusion[:, 2]) != 0:
                    Div_prec = confusion[2, 2]/np.sum(confusion[:, 2])
                else:
                    Div_prec = 0

                NB_rec = confusion[0, 0]/np.sum(confusion[0, :])
                Dif_rec = confusion[1, 1]/np.sum(confusion[1, :])
                Div_rec = confusion[2, 2]/np.sum(confusion[2, :])

                NB_prec_list.append(NB_prec)
                Dif_prec_list.append(Dif_prec)
                Div_prec_list.append(Div_prec)

                NB_rec_list.append(NB_rec)
                Dif_rec_list.append(Dif_rec)
                Div_rec_list.append(Div_rec)

                MacroAve_prec = (NB_prec+Dif_prec+Div_prec)/3
                MacroAve_rec = (NB_rec+Dif_rec+Div_rec)/3

                # print(NB_prec_list)
                # print(Dif_prec_list)
                # print(Div_prec_list)

                if np.isnan(MacroAve_prec) == False and np.isnan(MacroAve_rec) == False:
                    # print(MacroAve_prec)
                    # print(MacroAve_rec)
                    MacroF1 = stats.hmean([MacroAve_prec, MacroAve_rec])

                else:
                    MacroF1 = 0

                MacroF1_list.append(MacroF1)

            NB_prec_list_filename = "/labels/%s_NB_precision_list.pickle" % dir_base_train_test
            Dif_prec_list_filename = "/labels/%s_Dif_precision_list.pickle" % dir_base_train_test
            Div_prec_list_filename = "/labels/%s_Div_precision_list.pickle" % dir_base_train_test

            NB_rec_list_filename = "/labels/%s_NB_recall_list.pickle" % dir_base_train_test
            Dif_rec_list_filename = "/labels/%s_Dif_recall_list.pickle" % dir_base_train_test
            Div_rec_list_filename = "/labels/%s_Div_recall_list.pickle" % dir_base_train_test

            MacroF1_list_filename = "/labels/%s_MacroF1_list.pickle" % dir_base_train_test

            sutil.PickleDump(NB_prec_list, dir_path + NB_prec_list_filename)
            sutil.PickleDump(Dif_prec_list, dir_path + Dif_prec_list_filename)
            sutil.PickleDump(Div_prec_list, dir_path + Div_prec_list_filename)

            sutil.PickleDump(NB_rec_list, dir_path + NB_rec_list_filename)
            sutil.PickleDump(Dif_rec_list, dir_path + Dif_rec_list_filename)
            sutil.PickleDump(Div_rec_list, dir_path + Div_rec_list_filename)

            sutil.PickleDump(MacroF1_list, dir_path + MacroF1_list_filename)

            NB_rec_list_filename_txt = "test_NB_recall.txt"
            Dif_rec_list_filename_txt = "test_Del_recall.txt"
            Div_rec_list_filename_txt = "test_Div_recall.txt"

            MacroF1_list_filename_txt = "test_MacroF1Score.txt"

            sutil.SaveListText(
                NB_rec_list, save_dir_data_sample + NB_rec_list_filename_txt)
            sutil.SaveListText(
                Dif_rec_list, save_dir_data_sample + Dif_rec_list_filename_txt)
            sutil.SaveListText(
                Div_rec_list, save_dir_data_sample + Div_rec_list_filename_txt)

            sutil.SaveListText(
                MacroF1_list, save_dir_data_sample + MacroF1_list_filename_txt)

            NB_prec_list_test = copy.deepcopy(NB_prec_list)
            Dif_prec_list_test = copy.deepcopy(Dif_prec_list)
            Div_prec_list_test = copy.deepcopy(Div_prec_list)

            NB_rec_list_test = copy.deepcopy(NB_rec_list)
            Dif_rec_list_test = copy.deepcopy(Dif_rec_list)
            Div_rec_list_test = copy.deepcopy(Div_rec_list)

            MacroF1_list_test = copy.deepcopy(MacroF1_list)

            # For training data

            dir_base_train_test = "training/training_summary/training"
            NB_prec_list = []
            Dif_prec_list = []
            Div_prec_list = []

            NB_rec_list = []
            Dif_rec_list = []
            Div_rec_list = []

            MacroF1_list = []

            for index in index_log:
                filename_labels_test = filename_labels_train_pre + "%d.txt" % index
                confusion = np.loadtxt(dir_path + filename_labels_test)

                if np.sum(confusion[:, 0]) != 0:
                    NB_prec = confusion[0, 0]/np.sum(confusion[:, 0])
                else:
                    NB_prec = 0

                if np.sum(confusion[:, 1]) != 0:
                    Dif_prec = confusion[1, 1]/np.sum(confusion[:, 1])
                else:
                    Dif_prec = 0

                if np.sum(confusion[:, 2]) != 0:
                    Div_prec = confusion[2, 2]/np.sum(confusion[:, 2])
                else:
                    Div_prec = 0

                # NB_prec=confusion[0,0]/np.sum(confusion[:,0])
                # Dif_prec=confusion[1,1]/np.sum(confusion[:,1])
                # Div_prec=confusion[2,2]/np.sum(confusion[:,2])

                NB_rec = confusion[0, 0]/np.sum(confusion[0, :])
                Dif_rec = confusion[1, 1]/np.sum(confusion[1, :])
                Div_rec = confusion[2, 2]/np.sum(confusion[2, :])

                NB_prec_list.append(NB_prec)
                Dif_prec_list.append(Dif_prec)
                Div_prec_list.append(Div_prec)

                NB_rec_list.append(NB_rec)
                Dif_rec_list.append(Dif_rec)
                Div_rec_list.append(Div_rec)

                # print(NB_prec_list)
                # print(Dif_prec_list)
                # print(Div_prec_list)

                MacroAve_prec = (NB_prec+Dif_prec+Div_prec)/3
                MacroAve_rec = (NB_rec+Dif_rec+Div_rec)/3

                if np.isnan(MacroAve_prec) == False and np.isnan(MacroAve_rec) == False:
                    # print(MacroAve_prec)
                    # print(MacroAve_rec)
                    MacroF1 = stats.hmean([MacroAve_prec, MacroAve_rec])

                else:
                    MacroF1 = 0

                MacroF1_list.append(MacroF1)

            NB_prec_list_filename = "/labels/%s_NB_precision_list.pickle" % dir_base_train_test
            Dif_prec_list_filename = "/labels/%s_Dif_precision_list.pickle" % dir_base_train_test
            Div_prec_list_filename = "/labels/%s_Div_precision_list.pickle" % dir_base_train_test

            NB_rec_list_filename = "/labels/%s_NB_recall_list.pickle" % dir_base_train_test
            Dif_rec_list_filename = "/labels/%s_Dif_recall_list.pickle" % dir_base_train_test
            Div_rec_list_filename = "/labels/%s_Div_recall_list.pickle" % dir_base_train_test

            MacroF1_list_filename = "/labels/%s_MacroF1_list.pickle" % dir_base_train_test

            sutil.PickleDump(NB_prec_list, dir_path + NB_prec_list_filename)
            sutil.PickleDump(Dif_prec_list, dir_path + Dif_prec_list_filename)
            sutil.PickleDump(Div_prec_list, dir_path + Div_prec_list_filename)

            sutil.PickleDump(NB_rec_list, dir_path + NB_rec_list_filename)
            sutil.PickleDump(Dif_rec_list, dir_path + Dif_rec_list_filename)
            sutil.PickleDump(Div_rec_list, dir_path + Div_rec_list_filename)

            sutil.PickleDump(MacroF1_list, dir_path + MacroF1_list_filename)

            NB_rec_list_filename_txt = "train_NB_recall.txt"
            Dif_rec_list_filename_txt = "train_Del_recall.txt"
            Div_rec_list_filename_txt = "train_Div_recall.txt"

            MacroF1_list_filename_txt = "train_MacroF1Score.txt"

            sutil.SaveListText(
                NB_rec_list, save_dir_data_sample + NB_rec_list_filename_txt)
            sutil.SaveListText(
                Dif_rec_list, save_dir_data_sample + Dif_rec_list_filename_txt)
            sutil.SaveListText(
                Div_rec_list, save_dir_data_sample + Div_rec_list_filename_txt)

            sutil.SaveListText(
                MacroF1_list, save_dir_data_sample + MacroF1_list_filename_txt)

            NB_prec_list_train = copy.deepcopy(NB_prec_list)
            Dif_prec_list_train = copy.deepcopy(Dif_prec_list)
            Div_prec_list_train = copy.deepcopy(Div_prec_list)

            NB_rec_list_train = copy.deepcopy(NB_rec_list)
            Dif_rec_list_train = copy.deepcopy(Dif_rec_list)
            Div_rec_list_train = copy.deepcopy(Div_rec_list)

            MacroF1_list_train = copy.deepcopy(MacroF1_list)

            #acc_test = np.loadtxt(dir_path + filename_acc_test)
            #acc_train = np.loadtxt(dir_path + filename_acc_train)
            # x=range(len(acc_test))

            ax = fig.add_subplot(num_row, num_column, count)

            # ax.plot(x,acc_test)
            # ax.plot(x,acc_train)
            ax.plot(index_log, MacroF1_list_train,
                    c="cyan", label="TrainMacroF1")

            ax.plot(index_log, MacroF1_list_test,
                    c="green", label="TestMacroF1")

            MaxMacroF1_epoch = index_log[np.argmax(
                np.array(MacroF1_list_test))]

            argmax_F1 = np.argmax(np.array(MacroF1_list_test))

            MaxMacroF1 = np.max(np.array(MacroF1_list_test))

            print("MaxF1=%f at epoch %d" % (MaxMacroF1, MaxMacroF1_epoch))

            sutil.PickleDump(MaxMacroF1_epoch,  dir_path +
                             filename_MaxMacroF1_epoch)
            sutil.PickleDump(MaxMacroF1,  dir_path + filename_MaxMacroF1)

            sutil.SaveText(MaxMacroF1_epoch,  save_dir_data_sample +
                           filename_MaxMacroF1_epoch_txt)

            AUC_time_data = sutil.PickleLoad(dir_path + filename_AUC_time)

            AUCForMaxMacroF1 = [AUC_time_data[argmax_F1][0],
                                AUC_time_data[argmax_F1][1], AUC_time_data[argmax_F1][2]]
            print(AUCForMaxMacroF1)
            sutil.PickleDump(AUCForMaxMacroF1, dir_path +
                             filename_AUCForMaxMacroF1)

            # ax.plot(x,acc_train)

            ax.axvline(x=MaxMacroF1_epoch, ls="-", lw=0.5,
                       c="b", label="MaxTestMacroF1")
            # ax.axvline(x=Max_AUC_epoch,ls="-",lw=0.5,c="r")

            ax.plot(index_log, NB_rec_list_test,
                    c=cmap[0], lw=1, label="TestNBRecall")
            ax.plot(index_log, Dif_rec_list_test,
                    c=cmap[1], lw=1, label="TestDifRecall")
            ax.plot(index_log, Div_rec_list_test,
                    c=cmap[2], lw=1, label="TestDivRecall")

            ax.plot(index_log, NB_rec_list_train,
                    c=cmap[0], lw=1, ls="--", label="TrainNBRecall")
            ax.plot(index_log, Dif_rec_list_train,
                    c=cmap[1], lw=1, ls="--", label="TrainDifRecall")
            ax.plot(index_log, Div_rec_list_train,
                    c=cmap[2], lw=1, ls="--", label="TrainDivRecall")

            # ax.set_xticks(x_list)
           # ax.set_xlim([0.0, 1.0])
            #ax.set_ylim([ymin, ymax])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("F1OrRecall")
            ax.set_title('%s\n%s' % (name, listname))
            # print(labels)
            # if i == 0:
            #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.legend(fontsize=5, loc='upper right')

            #plt.legend(celltype_list,loc="upper right",fontsize=legend_size,framealpha=0)

        count += 1

        # print(features_list)
    #ax.axhline(0.5, ls = "--", color = "k",lw=0.75)

figname = save_dir_figs + "%s.pdf" % fig_title
plt.savefig(figname, transparent=True)
figname_png = save_dir_figs + "%s.png" % fig_title
plt.savefig(figname_png, dpi=dpi, transparent=True)

plt.show()

plt.close()


# +
# Plot the time evolution of precision for each label and macro-F1 score.
# When precision is nan, we define F1 = 0.
# Save maximum macro-F1 score and epoch for the max value.
# the data points are saved as "/labels/test/test_summary/test_MacroF1_list.pickle" and "test_NB_precision_list.pickle".

fig_title = "Precision_MacroF1Score_time_evolution"

num_row = len(listname_list)
num_column = len(name_list)

# figx_time=3
# figy_time=3
fig = plt.figure(figsize=(figx_time*num_column, figy_time*num_row))

fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.subplots_adjust(hspace=fig_hspace, wspace=fig_wspace)


save_dir = work_dir + "/summary_%s/%s/" % (version, fig_title)
sutil.MakeDirs(save_dir)

save_dir_figs = save_dir + "figs/"
save_dir_data = save_dir + "data/"

sutil.MakeDirs(save_dir_figs)
sutil.MakeDirs(save_dir_data)

count = 1
for i, listname in enumerate(listname_list):

    path = work_dir + "/%s.txt" % listname
    #path = "./SelfInfo_list.txt"
    dir_list = sutil.ReadLinesText(path)

    dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

    for j, dir_path in enumerate(dir_list):

        name = name_list[j]
        save_dir_data_sample = save_dir_data + name + "/sample%d/" % i
        sutil.MakeDirs(save_dir_data_sample)

        dir_path = dir_path + "/"

        parameters_file_path = dir_path + filename_parameters
        param = sutil.PickleLoad(parameters_file_path)

        model_save_rate = param["model_save_rate"]
        epoch_total = param["epoch_total"]
        index_log = list(range(0, epoch_total, model_save_rate))
        index_log.append(epoch_total-1)
        epoch_list_filename_txt = "epoch_list.txt"
        sutil.SaveListText(index_log, save_dir_data_sample +
                           epoch_list_filename_txt)

        acc_test_file_path = dir_path + filename_acc_test

        if os.path.exists(acc_test_file_path) == True:

            filename_labels_test_pre = "/labels/test/test_summary/CorrectVsPredict/test_CorrectVsPredictAll_epoch="
            filename_labels_train_pre = "/labels/training/training_summary/CorrectVsPredict/training_CorrectVsPredictAll_epoch="

            # Calculate recall for test and training data for each label

            # For test data
            dir_base_train_test = "test/test_summary/test"

            NB_prec_list = []
            Dif_prec_list = []
            Div_prec_list = []

            NB_rec_list = []
            Dif_rec_list = []
            Div_rec_list = []

            MacroF1_list = []

            for index in index_log:
                filename_labels_test = filename_labels_test_pre + "%d.txt" % index
                confusion = np.loadtxt(dir_path + filename_labels_test)

                if np.sum(confusion[:, 0]) != 0:
                    NB_prec = confusion[0, 0]/np.sum(confusion[:, 0])
                else:
                    NB_prec = 0

                if np.sum(confusion[:, 1]) != 0:
                    Dif_prec = confusion[1, 1]/np.sum(confusion[:, 1])
                else:
                    Dif_prec = 0

                if np.sum(confusion[:, 2]) != 0:
                    Div_prec = confusion[2, 2]/np.sum(confusion[:, 2])
                else:
                    Div_prec = 0
                # NB_prec=confusion[0,0]/np.sum(confusion[:,0])
                # Dif_prec=confusion[1,1]/np.sum(confusion[:,1])
                # Div_prec=confusion[2,2]/np.sum(confusion[:,2])

                NB_rec = confusion[0, 0]/np.sum(confusion[0, :])
                Dif_rec = confusion[1, 1]/np.sum(confusion[1, :])
                Div_rec = confusion[2, 2]/np.sum(confusion[2, :])

                NB_prec_list.append(NB_prec)
                Dif_prec_list.append(Dif_prec)
                Div_prec_list.append(Div_prec)

                NB_rec_list.append(NB_rec)
                Dif_rec_list.append(Dif_rec)
                Div_rec_list.append(Div_rec)

                MacroAve_prec = (NB_prec+Dif_prec+Div_prec)/3
                MacroAve_rec = (NB_rec+Dif_rec+Div_rec)/3

                # print(NB_prec_list)
                # print(Dif_prec_list)
                # print(Div_prec_list)

                if np.isnan(MacroAve_prec) == False and np.isnan(MacroAve_rec) == False:
                    # print(MacroAve_prec)
                    # print(MacroAve_rec)
                    MacroF1 = stats.hmean([MacroAve_prec, MacroAve_rec])

                else:
                    MacroF1 = 0

                MacroF1_list.append(MacroF1)

            NB_prec_list_filename = "/labels/%s_NB_precision_list.pickle" % dir_base_train_test
            Dif_prec_list_filename = "/labels/%s_Dif_precision_list.pickle" % dir_base_train_test
            Div_prec_list_filename = "/labels/%s_Div_precision_list.pickle" % dir_base_train_test

            NB_rec_list_filename = "/labels/%s_NB_recall_list.pickle" % dir_base_train_test
            Dif_rec_list_filename = "/labels/%s_Dif_recall_list.pickle" % dir_base_train_test
            Div_rec_list_filename = "/labels/%s_Div_recall_list.pickle" % dir_base_train_test

            MacroF1_list_filename = "/labels/%s_MacroF1_list.pickle" % dir_base_train_test

            sutil.PickleDump(NB_prec_list, dir_path + NB_prec_list_filename)
            sutil.PickleDump(Dif_prec_list, dir_path + Dif_prec_list_filename)
            sutil.PickleDump(Div_prec_list, dir_path + Div_prec_list_filename)

            sutil.PickleDump(NB_rec_list, dir_path + NB_rec_list_filename)
            sutil.PickleDump(Dif_rec_list, dir_path + Dif_rec_list_filename)
            sutil.PickleDump(Div_rec_list, dir_path + Div_rec_list_filename)

            sutil.PickleDump(MacroF1_list, dir_path + MacroF1_list_filename)

            NB_prec_list_filename_txt = "test_NB_precision.txt"
            Dif_prec_list_filename_txt = "test_Del_precision.txt"
            Div_prec_list_filename_txt = "test_Div_precision.txt"

            MacroF1_list_filename_txt = "test_MacroF1Score.txt"

            sutil.SaveListText(
                NB_rec_list, save_dir_data_sample + NB_prec_list_filename_txt)
            sutil.SaveListText(
                Dif_rec_list, save_dir_data_sample + Dif_prec_list_filename_txt)
            sutil.SaveListText(
                Div_rec_list, save_dir_data_sample + Div_prec_list_filename_txt)

            sutil.SaveListText(
                MacroF1_list, save_dir_data_sample + MacroF1_list_filename_txt)

            NB_prec_list_test = copy.deepcopy(NB_prec_list)
            Dif_prec_list_test = copy.deepcopy(Dif_prec_list)
            Div_prec_list_test = copy.deepcopy(Div_prec_list)

            NB_rec_list_test = copy.deepcopy(NB_rec_list)
            Dif_rec_list_test = copy.deepcopy(Dif_rec_list)
            Div_rec_list_test = copy.deepcopy(Div_rec_list)

            MacroF1_list_test = copy.deepcopy(MacroF1_list)

            # For training data

            dir_base_train_test = "training/training_summary/training"
            NB_prec_list = []
            Dif_prec_list = []
            Div_prec_list = []

            NB_rec_list = []
            Dif_rec_list = []
            Div_rec_list = []

            MacroF1_list = []

            for index in index_log:
                filename_labels_test = filename_labels_train_pre + "%d.txt" % index
                confusion = np.loadtxt(dir_path + filename_labels_test)

                if np.sum(confusion[:, 0]) != 0:
                    NB_prec = confusion[0, 0]/np.sum(confusion[:, 0])
                else:
                    NB_prec = 0

                if np.sum(confusion[:, 1]) != 0:
                    Dif_prec = confusion[1, 1]/np.sum(confusion[:, 1])
                else:
                    Dif_prec = 0

                if np.sum(confusion[:, 2]) != 0:
                    Div_prec = confusion[2, 2]/np.sum(confusion[:, 2])
                else:
                    Div_prec = 0

                # NB_prec=confusion[0,0]/np.sum(confusion[:,0])
                # Dif_prec=confusion[1,1]/np.sum(confusion[:,1])
                # Div_prec=confusion[2,2]/np.sum(confusion[:,2])

                NB_rec = confusion[0, 0]/np.sum(confusion[0, :])
                Dif_rec = confusion[1, 1]/np.sum(confusion[1, :])
                Div_rec = confusion[2, 2]/np.sum(confusion[2, :])

                NB_prec_list.append(NB_prec)
                Dif_prec_list.append(Dif_prec)
                Div_prec_list.append(Div_prec)

                NB_rec_list.append(NB_rec)
                Dif_rec_list.append(Dif_rec)
                Div_rec_list.append(Div_rec)

                # print(NB_prec_list)
                # print(Dif_prec_list)
                # print(Div_prec_list)

                MacroAve_prec = (NB_prec+Dif_prec+Div_prec)/3
                MacroAve_rec = (NB_rec+Dif_rec+Div_rec)/3

                if np.isnan(MacroAve_prec) == False and np.isnan(MacroAve_rec) == False:
                    # print(MacroAve_prec)
                    # print(MacroAve_rec)
                    MacroF1 = stats.hmean([MacroAve_prec, MacroAve_rec])

                else:
                    MacroF1 = 0

                MacroF1_list.append(MacroF1)

            NB_prec_list_filename = "/labels/%s_NB_precision_list.pickle" % dir_base_train_test
            Dif_prec_list_filename = "/labels/%s_Dif_precision_list.pickle" % dir_base_train_test
            Div_prec_list_filename = "/labels/%s_Div_precision_list.pickle" % dir_base_train_test

            NB_rec_list_filename = "/labels/%s_NB_recall_list.pickle" % dir_base_train_test
            Dif_rec_list_filename = "/labels/%s_Dif_recall_list.pickle" % dir_base_train_test
            Div_rec_list_filename = "/labels/%s_Div_recall_list.pickle" % dir_base_train_test

            MacroF1_list_filename = "/labels/%s_MacroF1_list.pickle" % dir_base_train_test

            sutil.PickleDump(NB_prec_list, dir_path + NB_prec_list_filename)
            sutil.PickleDump(Dif_prec_list, dir_path + Dif_prec_list_filename)
            sutil.PickleDump(Div_prec_list, dir_path + Div_prec_list_filename)

            sutil.PickleDump(NB_rec_list, dir_path + NB_rec_list_filename)
            sutil.PickleDump(Dif_rec_list, dir_path + Dif_rec_list_filename)
            sutil.PickleDump(Div_rec_list, dir_path + Div_rec_list_filename)

            sutil.PickleDump(MacroF1_list, dir_path + MacroF1_list_filename)

            NB_prec_list_filename_txt = "train_NB_precision.txt"
            Dif_prec_list_filename_txt = "train_Del_precision.txt"
            Div_prec_list_filename_txt = "train_Div_precision.txt"

            MacroF1_list_filename_txt = "train_MacroF1Score.txt"

            sutil.SaveListText(
                NB_rec_list, save_dir_data_sample + NB_prec_list_filename_txt)
            sutil.SaveListText(
                Dif_rec_list, save_dir_data_sample + Dif_prec_list_filename_txt)
            sutil.SaveListText(
                Div_rec_list, save_dir_data_sample + Div_prec_list_filename_txt)

            sutil.SaveListText(
                MacroF1_list, save_dir_data_sample + MacroF1_list_filename_txt)

            NB_prec_list_train = copy.deepcopy(NB_prec_list)
            Dif_prec_list_train = copy.deepcopy(Dif_prec_list)
            Div_prec_list_train = copy.deepcopy(Div_prec_list)

            NB_rec_list_train = copy.deepcopy(NB_rec_list)
            Dif_rec_list_train = copy.deepcopy(Dif_rec_list)
            Div_rec_list_train = copy.deepcopy(Div_rec_list)

            MacroF1_list_train = copy.deepcopy(MacroF1_list)

            #acc_test = np.loadtxt(dir_path + filename_acc_test)
            #acc_train = np.loadtxt(dir_path + filename_acc_train)
            # x=range(len(acc_test))

            ax = fig.add_subplot(num_row, num_column, count)

            # ax.plot(x,acc_test)
            # ax.plot(x,acc_train)
            ax.plot(index_log, MacroF1_list_train,
                    c="cyan", label="TrainMacroF1")

            ax.plot(index_log, MacroF1_list_test,
                    c="green", label="TestMacroF1")

            # print(MacroF1_list_train[100])
            # print(MacroF1_list[100])

            MaxMacroF1_epoch = index_log[np.argmax(
                np.array(MacroF1_list_test))]

            MaxMacroF1 = np.max(np.array(MacroF1_list_test))

            print("MaxF1=%f at epoch %d" % (MaxMacroF1, MaxMacroF1_epoch))

            sutil.PickleDump(MaxMacroF1_epoch,  dir_path +
                             filename_MaxMacroF1_epoch)
            sutil.PickleDump(MaxMacroF1,  dir_path + filename_MaxMacroF1)

            sutil.SaveText(MaxMacroF1_epoch,  save_dir_data_sample +
                           filename_MaxMacroF1_epoch_txt)

            # ax.plot(x,acc_train)

            ax.axvline(x=MaxMacroF1_epoch, ls="-", lw=0.5,
                       c="b", label="MaxTestMacroF1")
            # ax.axvline(x=Max_AUC_epoch,ls="-",lw=0.5,c="r")

            ax.plot(index_log, NB_prec_list_test,
                    c=cmap[0], lw=1, label="TestNBPrec")
            ax.plot(index_log, Dif_prec_list_test,
                    c=cmap[1], lw=1, label="TestDifPrec")
            ax.plot(index_log, Div_prec_list_test,
                    c=cmap[2], lw=1, label="TestDivPrec")

            ax.plot(index_log, NB_prec_list_train,
                    c=cmap[0], lw=1, ls="--", label="TrainNBPrec")
            ax.plot(index_log, Dif_prec_list_train,
                    c=cmap[1], lw=1, ls="--", label="TrainDifPrec")
            ax.plot(index_log, Div_prec_list_train,
                    c=cmap[2], lw=1, ls="--", label="TrainDivPrec")

            # ax.set_xticks(x_list)
           # ax.set_xlim([0.0, 1.0])
            #ax.set_ylim([ymin, ymax])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("F1OrPrecision")
            ax.set_title('%s\n%s' % (name, listname))
            # print(labels)
            # if i == 0:
            #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.legend(fontsize=5, loc='upper right')

            #plt.legend(celltype_list,loc="upper right",fontsize=legend_size,framealpha=0)

        count += 1

        # print(features_list)
    #ax.axhline(0.5, ls = "--", color = "k",lw=0.75)

figname = save_dir_figs + "%s.pdf" % fig_title
plt.savefig(figname, transparent=True)
figname_png = save_dir_figs + "%s.png" % fig_title
plt.savefig(figname_png, dpi=dpi, transparent=True)

plt.show()

plt.close()

# -


# +
# Plot the time evolution of AUC for each label and mean AUC.
# the data points are saved as "/labels/test/test_summary/test_Div_CorrectCellNumber_list.pickle".

fig_title = "AUC_time_evolution"

num_row = len(listname_list)
num_column = len(name_list)

# figx_time=3
# figy_time=3
fig = plt.figure(figsize=(figx_time*num_column, figy_time*num_row))

fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.subplots_adjust(hspace=fig_hspace, wspace=fig_wspace)


save_dir = work_dir + "/summary_%s/%s/" % (version, fig_title)
sutil.MakeDirs(save_dir)

save_dir_figs = save_dir + "figs/"
save_dir_data = save_dir + "data/"

sutil.MakeDirs(save_dir_figs)
sutil.MakeDirs(save_dir_data)

count = 1
for i, listname in enumerate(listname_list):

    path = work_dir + "/%s.txt" % listname
    #path = "./SelfInfo_list.txt"
    dir_list = sutil.ReadLinesText(path)

    dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

    for j, dir_path in enumerate(dir_list):

        name = name_list[j]
        save_dir_data_sample = save_dir_data + name + "/sample%d/" % i
        sutil.MakeDirs(save_dir_data_sample)

        dir_path = dir_path + "/"

        parameters_file_path = dir_path + filename_parameters
        param = sutil.PickleLoad(parameters_file_path)

        model_save_rate = param["model_save_rate"]
        epoch_total = param["epoch_total"]
        index_log = list(range(0, epoch_total, model_save_rate))
        index_log.append(epoch_total-1)
        epoch_list_filename_txt = "epoch_list.txt"
        sutil.SaveListText(index_log, save_dir_data_sample +
                           epoch_list_filename_txt)

        acc_test_file_path = dir_path + filename_acc_test

        if os.path.exists(acc_test_file_path) == True:

            acc_test_history = np.loadtxt(acc_test_file_path)
            acc_test_history_target = acc_test_history[index_log]

            max_acc = np.max(acc_test_history_target)
            argmax_acc = np.argmax(acc_test_history_target)

            epoch_maxacc = index_log[argmax_acc]

            print("MaxACC%f at epoch %d" % (max_acc, epoch_maxacc))

            Max_AUC_epoch = sutil.PickleLoad(dir_path + filename_Max_AUC_epoch)

            AUC_time_list = sutil.PickleLoad(dir_path + filename_AUC_time)

            # print(AUC_time_list)

            AUC_NB_list = []
            AUC_Del_list = []
            AUC_Div_list = []
            AUC_mean_list = []
            for AUC_index, AUC_time in enumerate(AUC_time_list):
                AUC_NB_list.append(AUC_time[0])
                AUC_Del_list.append(AUC_time[1])
                AUC_Div_list.append(AUC_time[2])
                AUC_mean_list.append(
                    (AUC_time[0] + AUC_time[1] + AUC_time[2])/3)

            AUC_NB_list_filename_txt = "test_NB_AUC.txt"
            AUC_Del_list_filename_txt = "test_Del_AUC.txt"
            AUC_Div_list_filename_txt = "test_Div_AUC.txt"

            AUC_mean_list_filename_txt = "test_mean_AUC.txt"

            sutil.SaveListText(
                AUC_NB_list, save_dir_data_sample + AUC_NB_list_filename_txt)
            sutil.SaveListText(
                AUC_Del_list, save_dir_data_sample + AUC_Del_list_filename_txt)
            sutil.SaveListText(
                AUC_Div_list, save_dir_data_sample + AUC_Div_list_filename_txt)

            sutil.SaveListText(
                AUC_mean_list, save_dir_data_sample + AUC_mean_list_filename_txt)

            ax = fig.add_subplot(num_row, num_column, count)

            MaxMacroF1_epoch = sutil.PickleLoad(
                dir_path + filename_MaxMacroF1_epoch)
            ax.axvline(x=MaxMacroF1_epoch, ls="-", lw=0.5,
                       c="b", label="MaxTestMacroF1")

            sutil.SaveText(MaxMacroF1_epoch,  save_dir_data_sample +
                           filename_MaxMacroF1_epoch_txt)

            ax.axvline(x=Max_AUC_epoch, ls="--", lw=0.5,
                       c="r", label="MaxTestAUC")

            sutil.SaveText(Max_AUC_epoch,  save_dir_data_sample +
                           filename_Max_AUC_epoch_txt)

            ax.plot(index_log, AUC_NB_list, c=cmap[0], lw=1, label="NB")
            ax.plot(index_log, AUC_Del_list, c=cmap[1], lw=1, label="Dif")
            ax.plot(index_log, AUC_Div_list, c=cmap[2], lw=1, label="Div")
            ax.plot(index_log, AUC_mean_list, c="k", lw=1, label="Mean")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("AUC")
            ax.set_title('%s\n%s' % (name, listname))
            # print(labels)
            # if i == 0:
            #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.legend(fontsize=5, loc='upper right')

        #plt.legend(celltype_list,loc="upper right",fontsize=legend_size,framealpha=0)

        count += 1

        # print(features_list)
    #ax.axhline(0.5, ls = "--", color = "k",lw=0.75)

figname = save_dir_figs + "%s.pdf" % fig_title
plt.savefig(figname, transparent=True)
figname_png = save_dir_figs + "%s.png" % fig_title
plt.savefig(figname_png, dpi=dpi, transparent=True)

plt.show()

plt.close()


# -
# Plot time evolution of max of absolute values of weight and bias of the model
# Default: no plot by setting weight_plot==0
if weight_plot == 1:
    num_row = len(listname_list)
    num_column = len(name_list)

    fig = plt.figure(figsize=(figx_time*num_column, figy_time*num_row))

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.subplots_adjust(hspace=fig_hspace, wspace=fig_wspace)

    save_dir = work_dir + "/summary_%s/WeightCurve/" % (version)
    sutil.MakeDirs(save_dir)

    save_dir_figs = save_dir + "figs/"
    save_dir_data = save_dir + "data/"

    sutil.MakeDirs(save_dir_figs)
    sutil.MakeDirs(save_dir_data)

    count = 1
    for i, listname in enumerate(listname_list):

        path = work_dir + "/%s.txt" % listname
        #path = "./SelfInfo_list.txt"
        dir_list = sutil.ReadLinesText(path)

        dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

        for j, dir_path in enumerate(dir_list):

            name = name_list[j]
            save_dir_data_sample = save_dir_data + name + "/sample%d/" % i
            sutil.MakeDirs(save_dir_data_sample)

            dir_path = dir_path + "/"

            model_dirname_pre = "model/model_epoch="

            filename_parameters = "parameters.pickle"

            parameters_file_path = dir_path + filename_parameters
            param = sutil.PickleLoad(parameters_file_path)

            hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature = LoadTrainingParameters(
                param)

            index_log = list(range(0, epoch_total, model_save_rate))
            index_log.append(epoch_total-1)

            if architecture == "SP":

                model = gnn_models.CellFateNet(input_size, in_feats, hid_node, 3, feature, time_list, p_hidden, in_plane, NoSelfInfo,
                                               feature_self, n_layers, skip, average_switch, edge_switch, input_size_edge, feature_edge_concat, reg, norm_final)
                model_ver = model.version

            if architecture == "NSP":
                model = gnn_models.CellFateNetTimeReversal(input_size, in_feats, hid_node, 3, feature, time_list, p_hidden, in_plane,
                                                           NoSelfInfo, feature_self, n_layers, skip, edge_switch, input_size_edge, feature_edge_concat, average_switch)
                model_ver = model.version

            # for epoch in index_log:
            abs_max_list_all_bias = []
            abs_max_list_all_weight = []
            for epoch in index_log:
                # for epoch in [1999]:

                model_path = dir_path + model_dirname_pre + "%012d.pt" % epoch
                # print(model_path)
                #model = model.to(device)

                if gpu != -1:
                    model.load_state_dict(
                        th.load(model_path))  # if load by GPU
                else:
                    model.load_state_dict(
                        th.load(model_path, map_location=th.device('cpu')))  # if load by CPU

                # print(model)
                # print(model.state_dict().keys())
                parameter_name_list = list(model.state_dict().keys())
                # print(parameter_name_list)

                # Omit encoder
                parameter_name_list = parameter_name_list[n_layers*2:]
                # model.parameters()
                abs_max_list_bias = []
                abs_max_list_weight = []

                key = "bias"
                parameter_name_list_bias = [
                    s for s in parameter_name_list if key in s]

                for parameter_name_bias in parameter_name_list_bias:
                    # print(parameter_name)
                    parameters = model.state_dict()[parameter_name_bias].to(
                        "cpu").detach().numpy().copy()
                    # print(parameters.shape)
                    abs_max_bias = np.max(np.abs(parameters))
                    abs_max_list_bias.append(abs_max_bias)

                # max(abs_max_list)

                abs_max_list_all_bias.append(max(abs_max_list_bias))

                key = "weight"
                parameter_name_list_weight = [
                    s for s in parameter_name_list if key in s]

                for parameter_name_weight in parameter_name_list_weight:
                    # print(parameter_name)
                    parameters = model.state_dict()[parameter_name_weight].to(
                        "cpu").detach().numpy().copy()
                    # print(parameters.shape)
                    abs_max_weight = np.max(np.abs(parameters))
                    abs_max_list_weight.append(abs_max_weight)

                # max(abs_max_list)

                abs_max_list_all_weight.append(max(abs_max_list_weight))

            ax = fig.add_subplot(num_row, num_column, count)

            # ax.set_xticks(parameter_name_list)
            ax.plot(index_log, abs_max_list_all_bias, label="bias")

            ax.plot(index_log, abs_max_list_all_weight, label="weight")

            # ax.set_xlim([0.0, 1.0])
            #ax.set_ylim([ymin, ymax])
            # ax.set_xlabel(xlabel)
            # ax.set_ylabel(ylabel)

            ax.set_title('%s\n%s' % (name, listname))
            # ax.set_title("%s\n"%(name))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Max of abs of weights")
            # print(labels)
            # if i == 0:
            plt.legend(loc='upper left')

            #plt.legend(celltype_list,loc="upper right",fontsize=legend_size,framealpha=0)

            #ax.axhline(0.5, ls = "--", color = "k",lw=0.75)

            sutil.PickleDump(parameter_name_list, save_dir_data +
                             "layer_name_list_%d.pickle" % count)
            sutil.PickleDump(index_log, save_dir_data +
                             "index_log_%d.pickle" % count)
            sutil.PickleDump(abs_max_list_all_weight, save_dir_data +
                             "abs_max_list_all_weight_%d.pickle" % count)
            sutil.PickleDump(abs_max_list_all_bias, save_dir_data +
                             "abs_max_list_all_bias_%d.pickle" % count)

            count += 1

    figname = save_dir_figs + "Max_abs_weight.pdf"
    plt.savefig(figname, transparent=True)
    figname_png = save_dir_figs + "Max_abs_weight.png"
    plt.savefig(figname_png, dpi=dpi, transparent=True)

    plt.show()

    plt.close()

    del model

    th.cuda.empty_cache()


# +
# Plot the time evolution of the number of correctly-predicted cells for each label.
# the data points are saved as "/labels/test/test_summary/test_Div_CorrectCellNumber_list.pickle".

if NumberOfCorrectPrediction_plot == 1:
    fig_title = "Number_of_correct_prediction_time_evolution"

    num_row = len(listname_list)
    num_column = len(name_list)

    # figx_time=3
    # figy_time=3
    fig = plt.figure(figsize=(figx_time*num_column, figy_time*num_row))

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.subplots_adjust(hspace=fig_hspace, wspace=fig_wspace)

    save_dir = work_dir + "/summary_%s/%s/" % (version, fig_title)
    sutil.MakeDirs(save_dir)

    save_dir_figs = save_dir + "figs/"
    save_dir_data = save_dir + "data/"

    sutil.MakeDirs(save_dir_figs)
    sutil.MakeDirs(save_dir_data)

    count = 1
    for i, listname in enumerate(listname_list):

        path = work_dir + "/%s.txt" % listname
        #path = "./SelfInfo_list.txt"
        dir_list = sutil.ReadLinesText(path)

        dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

        for j, dir_path in enumerate(dir_list):

            name = name_list[j]
            save_dir_data_sample = save_dir_data + name + "/sample%d/" % i
            sutil.MakeDirs(save_dir_data_sample)

            dir_path = dir_path + "/"

            parameters_file_path = dir_path + filename_parameters
            param = sutil.PickleLoad(parameters_file_path)

            model_save_rate = param["model_save_rate"]
            epoch_total = param["epoch_total"]
            index_log = list(range(0, epoch_total, model_save_rate))
            index_log.append(epoch_total-1)
            epoch_list_filename_txt = "epoch_list.txt"
            sutil.SaveListText(
                index_log, save_dir_data_sample + epoch_list_filename_txt)

            acc_test_file_path = dir_path + filename_acc_test
            if os.path.exists(acc_test_file_path) == True:
                acc_test_history = np.loadtxt(acc_test_file_path)
                acc_test_history_target = acc_test_history[index_log]

                max_acc = np.max(acc_test_history_target)
                argmax_acc = np.argmax(acc_test_history_target)

                epoch_maxacc = index_log[argmax_acc]

                print("MaxACC%f at epoch %d" % (max_acc, epoch_maxacc))

                Max_AUC_epoch = sutil.PickleLoad(
                    dir_path + filename_Max_AUC_epoch)

                filename_labels_test_pre = "/labels/test/test_summary/CorrectVsPredict/test_CorrectVsPredictAll_epoch="
                filename_labels_train_pre = "/labels/training/training_summary/CorrectVsPredict/training_CorrectVsPredictAll_epoch="

                NB_rate_list = []
                Dif_rate_list = []
                Div_rate_list = []
                for index in index_log:
                    filename_labels_test = filename_labels_test_pre + "%d.txt" % index
                    confusion = np.loadtxt(dir_path + filename_labels_test)
                    NB_rate = confusion[0, 0]  # /np.sum(confusion[0,:])
                    Dif_rate = confusion[1, 1]  # /np.sum(confusion[1,:])
                    Div_rate = confusion[2, 2]  # /np.sum(confusion[2,:])

                    NB_rate_list.append(NB_rate)
                    Dif_rate_list.append(Dif_rate)
                    Div_rate_list.append(Div_rate)

                NB_rate_list_filename = "/labels/test/test_summary/test_NB_CorrectCellNumber_list.pickle"
                Dif_rate_list_filename = "/labels/test/test_summary/test_Dif_CorrectCellNumber_list.pickle"
                Div_rate_list_filename = "/labels/test/test_summary/test_Div_CorrectCellNumber_list.pickle"

                sutil.PickleDump(NB_rate_list, dir_path +
                                 NB_rate_list_filename)
                sutil.PickleDump(Dif_rate_list, dir_path +
                                 Dif_rate_list_filename)
                sutil.PickleDump(Div_rate_list, dir_path +
                                 Div_rate_list_filename)

                NB_rate_list_train = []
                Dif_rate_list_train = []
                Div_rate_list_train = []
                for index in index_log:
                    filename_labels_train = filename_labels_train_pre + "%d.txt" % index
                    confusion = np.loadtxt(dir_path + filename_labels_train)
                    NB_rate_train = confusion[0, 0]  # /np.sum(confusion[0,:])
                    Dif_rate_train = confusion[1, 1]  # /np.sum(confusion[1,:])
                    Div_rate_train = confusion[2, 2]  # /np.sum(confusion[2,:])

                    NB_rate_list_train.append(NB_rate_train)
                    Dif_rate_list_train.append(Dif_rate_train)
                    Div_rate_list_train.append(Div_rate_train)

                NB_rate_list_filename_train = "/labels/test/test_summary/train_NB_CorrectCellNumber_list.pickle"
                Dif_rate_list_filename_train = "/labels/test/test_summary/train_Dif_CorrectCellNumber_list.pickle"
                Div_rate_list_filename_train = "/labels/test/test_summary/train_Div_CorrectCellNumber_list.pickle"

                sutil.PickleDump(NB_rate_list_train, dir_path +
                                 NB_rate_list_filename_train)
                sutil.PickleDump(Dif_rate_list_train,
                                 dir_path + Dif_rate_list_filename_train)
                sutil.PickleDump(Div_rate_list_train,
                                 dir_path + Div_rate_list_filename_train)

                #acc_test = np.loadtxt(dir_path + filename_acc_test)
                acc_train = np.loadtxt(dir_path + filename_acc_train)
                # x=range(len(acc_test))

                ax = fig.add_subplot(num_row, num_column, count)

                # ax.plot(x,acc_test)
                # ax.plot(x,acc_train)
                # ax.plot(index_log,acc_train[index_log],c="cyan",label="TrainACC")

                # ax.plot(index_log,acc_test_history_target,c="green",label="TestACC")
                # ax.plot(x,acc_train)

                MaxMacroF1_epoch = sutil.PickleLoad(
                    dir_path + filename_MaxMacroF1_epoch)
                ax.axvline(x=MaxMacroF1_epoch, ls="-", lw=0.5,
                           c="b", label="MaxTestMacroF1")

                # ax.axvline(x=Max_AUC_epoch,ls="-",lw=0.5,c="r")

                ax.plot(index_log, NB_rate_list,
                        c=cmap[0], lw=1, label="TestNB")
                ax.plot(index_log, Dif_rate_list,
                        c=cmap[1], lw=1, label="TestDif")
                ax.plot(index_log, Div_rate_list,
                        c=cmap[2], lw=1, label="TestDiv")

                ax.plot(index_log, NB_rate_list_train,
                        c=cmap[0], lw=1, ls="--", label="TrainNB")
                ax.plot(index_log, Dif_rate_list_train,
                        c=cmap[1], lw=1, ls="--", label="TrainDif")
                ax.plot(index_log, Div_rate_list_train,
                        c=cmap[2], lw=1, ls="--", label="TrainDiv")

                # ax.set_xticks(x_list)
               # ax.set_xlim([0.0, 1.0])
                #ax.set_ylim([ymin, ymax])
                ax.set_xlabel("Epoch")
                ax.set_ylabel("# of correct cells")
                ax.set_title('%s\n%s' % (name, listname))
                # print(labels)
                # if i == 0:
                #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.legend(fontsize=5, loc='upper right')

            #plt.legend(celltype_list,loc="upper right",fontsize=legend_size,framealpha=0)

            count += 1

            # print(features_list)
        #ax.axhline(0.5, ls = "--", color = "k",lw=0.75)

    figname = save_dir_figs + "%s.pdf" % fig_title
    plt.savefig(figname, transparent=True)
    figname_png = save_dir_figs + "%s.png" % fig_title
    plt.savefig(figname_png, dpi=dpi, transparent=True)

    plt.show()

    plt.close()


# +
# Plot the time evolution of recall curves for each label and balanced ACC weighted (different from balanced ACC) curves.
# Load balanced ACC history and epoch for Max AUC from the results of the training.
# Save maximum balanced ACC and epoch for the max value.
# the data points are saved as "/labels/test/test_summary/test_NB_rate_list.pickle" and "test_BalancedACCWeighted_list.pickle".

if balancedACCWeighted_plot == 1:
    fig_title = "Recall_BalancedACCWeighted_time_evolution"

    num_row = len(listname_list)
    num_column = len(name_list)

    # figx_time=3
    # figy_time=3
    fig = plt.figure(figsize=(figx_time*num_column, figy_time*num_row))

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.subplots_adjust(hspace=fig_hspace, wspace=fig_wspace)

    save_dir = work_dir + "/summary_%s/%s/" % (version, fig_title)
    sutil.MakeDirs(save_dir)

    save_dir_figs = save_dir + "figs/"
    save_dir_data = save_dir + "data/"

    sutil.MakeDirs(save_dir_figs)
    sutil.MakeDirs(save_dir_data)

    count = 1
    for i, listname in enumerate(listname_list):

        path = work_dir + "/%s.txt" % listname
        #path = "./SelfInfo_list.txt"
        dir_list = sutil.ReadLinesText(path)

        dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

        for j, dir_path in enumerate(dir_list):

            name = name_list[j]
            save_dir_data_sample = save_dir_data + name + "/sample%d/" % i
            sutil.MakeDirs(save_dir_data_sample)

            dir_path = dir_path + "/"

            parameters_file_path = dir_path + filename_parameters
            param = sutil.PickleLoad(parameters_file_path)

            # print(param)
            model_save_rate = param["model_save_rate"]
            epoch_total = param["epoch_total"]
            index_log = list(range(0, epoch_total, model_save_rate))
            index_log.append(epoch_total-1)

            acc_test_file_path = dir_path + filename_acc_test

            if os.path.exists(acc_test_file_path) == True:

                filename_labels_test_pre = "/labels/test/test_summary/CorrectVsPredict/test_CorrectVsPredictAll_epoch="
                filename_labels_train_pre = "/labels/training/training_summary/CorrectVsPredict/training_CorrectVsPredictAll_epoch="

                # Calculate recall for test and training data for each label
                NB_rate_list = []
                Dif_rate_list = []
                Div_rate_list = []

                #weight_list = []

                BalancedACCWeighted_list = []
                for index in index_log:
                    filename_labels_test = filename_labels_test_pre + "%d.txt" % index
                    confusion = np.loadtxt(dir_path + filename_labels_test)
                    NB_rate = confusion[0, 0]/np.sum(confusion[0, :])
                    Dif_rate = confusion[1, 1]/np.sum(confusion[1, :])
                    Div_rate = confusion[2, 2]/np.sum(confusion[2, :])

                    NB_rate_list.append(NB_rate)
                    Dif_rate_list.append(Dif_rate)
                    Div_rate_list.append(Div_rate)

                    weight_tmp = [np.sum(confusion[0, :])/np.sum(confusion), np.sum(
                        confusion[1, :])/np.sum(confusion), np.sum(confusion[2, :])/np.sum(confusion)]

                    # weight_list.append(weight_tmp)

                    BalancedACCWeighted = (
                        NB_rate*weight_tmp[0] + Dif_rate*weight_tmp[1] + Div_rate*weight_tmp[2])/3

                    BalancedACCWeighted_list.append(BalancedACCWeighted)

                NB_rate_list_filename = "/labels/test/test_summary/test_NB_rate_list.pickle"
                Dif_rate_list_filename = "/labels/test/test_summary/test_Dif_rate_list.pickle"
                Div_rate_list_filename = "/labels/test/test_summary/test_Div_rate_list.pickle"

                BalancedACCWeighted_list_filename = "/labels/test/test_summary/test_BalancedACCWeighted_list.pickle"

                sutil.PickleDump(NB_rate_list, dir_path +
                                 NB_rate_list_filename)
                sutil.PickleDump(Dif_rate_list, dir_path +
                                 Dif_rate_list_filename)
                sutil.PickleDump(Div_rate_list, dir_path +
                                 Div_rate_list_filename)

                sutil.PickleDump(BalancedACCWeighted_list,
                                 dir_path + BalancedACCWeighted_list_filename)

                NB_rate_list_train = []
                Dif_rate_list_train = []
                Div_rate_list_train = []
                #weight_list_train = []

                BalancedACCWeighted_list_train = []
                for index in index_log:
                    filename_labels_train = filename_labels_train_pre + "%d.txt" % index
                    confusion = np.loadtxt(dir_path + filename_labels_train)
                    NB_rate_train = confusion[0, 0]/np.sum(confusion[0, :])
                    Dif_rate_train = confusion[1, 1]/np.sum(confusion[1, :])
                    Div_rate_train = confusion[2, 2]/np.sum(confusion[2, :])

                    NB_rate_list_train.append(NB_rate_train)
                    Dif_rate_list_train.append(Dif_rate_train)
                    Div_rate_list_train.append(Div_rate_train)

                    weight_tmp_train = [np.sum(confusion[0, :])/np.sum(confusion), np.sum(
                        confusion[1, :])/np.sum(confusion), np.sum(confusion[2, :])/np.sum(confusion)]

                    # weight_list_train.append(weight_tmp_train)

                    BalancedACCWeighted_train = (
                        NB_rate_train*weight_tmp_train[0] + Dif_rate_train*weight_tmp_train[1] + Div_rate_train*weight_tmp_train[2])/3

                    BalancedACCWeighted_list_train.append(
                        BalancedACCWeighted_train)

                NB_rate_list_filename_train = "/labels/training/training_summary/train_NB_rate_list.pickle"
                Dif_rate_list_filename_train = "/labels/training/training_summary/train_Dif_rate_list.pickle"
                Div_rate_list_filename_train = "/labels/training/training_summary/train_Div_rate_list.pickle"

                BalancedACCWeighted_list_filename_train = "/labels/training/training_summary/train_BalancedACCWeighted_list.pickle"

                sutil.PickleDump(NB_rate_list_train, dir_path +
                                 NB_rate_list_filename_train)
                sutil.PickleDump(Dif_rate_list_train,
                                 dir_path + Dif_rate_list_filename_train)
                sutil.PickleDump(Div_rate_list_train,
                                 dir_path + Div_rate_list_filename_train)

                sutil.PickleDump(BalancedACCWeighted_list_train,
                                 dir_path + BalancedACCWeighted_list_filename_train)

                #acc_test = np.loadtxt(dir_path + filename_acc_test)
                acc_train = np.loadtxt(dir_path + filename_acc_train)
                # x=range(len(acc_test))

                ax = fig.add_subplot(num_row, num_column, count)

                # ax.plot(x,acc_test)
                # ax.plot(x,acc_train)
                ax.plot(index_log, BalancedACCWeighted_list_train,
                        c="cyan", label="TrainBalancedACCWeighted")

                ax.plot(index_log, BalancedACCWeighted_list,
                        c="green", label="TestBalancedACCWeighted")

                MaxBalancedACCWeighted_epoch = index_log[np.argmax(
                    np.array(BalancedACCWeighted_list))]

                MaxBalancedACCWeighted = np.max(
                    np.array(BalancedACCWeighted_list))

                # ax.plot(x,acc_train)

                ax.axvline(x=MaxBalancedACCWeighted_epoch, ls="-",
                           lw=0.5, c="b", label="MaxTestBalancedACCWeighted")
                # ax.axvline(x=Max_AUC_epoch,ls="-",lw=0.5,c="r")

                ax.plot(index_log, NB_rate_list,
                        c=cmap[0], lw=1, label="TestNBRecall")
                ax.plot(index_log, Dif_rate_list,
                        c=cmap[1], lw=1, label="TestDifRecall")
                ax.plot(index_log, Div_rate_list,
                        c=cmap[2], lw=1, label="TestDivRecall")

                ax.plot(index_log, NB_rate_list_train,
                        c=cmap[0], lw=1, ls="--", label="TrainNBRecall")
                ax.plot(index_log, Dif_rate_list_train,
                        c=cmap[1], lw=1, ls="--", label="TrainDifRecall")
                ax.plot(index_log, Div_rate_list_train,
                        c=cmap[2], lw=1, ls="--", label="TrainDivRecall")

                # ax.set_xticks(x_list)
               # ax.set_xlim([0.0, 1.0])
                #ax.set_ylim([ymin, ymax])
                ax.set_xlabel("Epoch")
                ax.set_ylabel("RecallOrACC")
                ax.set_title('%s\n%s' % (name, listname))
                # print(labels)
                # if i == 0:
                #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.legend(fontsize=5, loc='upper right')

                #plt.legend(celltype_list,loc="upper right",fontsize=legend_size,framealpha=0)

            count += 1

            # print(features_list)
        #ax.axhline(0.5, ls = "--", color = "k",lw=0.75)

    figname = save_dir_figs + "%s.pdf" % fig_title
    plt.savefig(figname, transparent=True)
    figname_png = save_dir_figs + "%s.png" % fig_title
    plt.savefig(figname_png, dpi=dpi, transparent=True)

    plt.show()

    plt.close()

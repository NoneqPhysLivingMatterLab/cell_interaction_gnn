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

""" Confusion matrix averaged over samples"""

import matplotlib.pyplot as plt
import numpy as np
from functions import system_utility as sutil
import os
import sys
version = 1
print("version=%d" % version)


celltype_list = ["NB", "Del", "Div"]
dict_list = [0, 1, 2]

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


data_type = base_name
version = "Performance-%s-%.2f" % (data_type, axis_width)

listname_list = ["%s-sample%d" % (data_type, i) for i in index_list]


title_ave = "%d-Average:%s" % (len(listname_list), data_type)

n_sample = len(listname_list)
n_xlabel = len(name_list)

n_legend = len(dict_list)


# -

def AverageConfusionMatrixOverSamples(filename_Max_epoch, listname_list, name_list, data_type_name, version, work_dir):
    """
    Average confusion matrix over samples.
    """

    # Plot AUCs for Max AUC models for each sample

    for sample, listname in enumerate(listname_list):

        path = work_dir + "/%s.txt" % listname
        dir_list = sutil.ReadLinesText(path)

        dir_list = sutil.RemoveCyberduckPrefixSuffix(dir_list)

        for dir_idx, dir_path in enumerate(dir_list):

            model_type = name_list[dir_idx]

            save_dir = work_dir + \
                "/summary_%s/%s/%s/" % (version, data_type_name, model_type)
            #save_dir_figs = save_dir + "figs/"
            save_dir_data = save_dir + "data/"

            # sutil.MakeDirs(save_dir_figs)
            sutil.MakeDirs(save_dir_data)

            dir_path = dir_path + "/"

            filepath_Max_epoch = dir_path + filename_Max_epoch
            Max_epoch = sutil.PickleLoad(filepath_Max_epoch)

            filepath_confusion_test = dir_path + \
                "labels/test/test_summary/CorrectVsPredict/test_CorrectVsPredictAll_epoch=%d.txt" % Max_epoch
            filepath_confusion_train = dir_path + \
                "labels/training/training_summary/CorrectVsPredict/training_CorrectVsPredictAll_epoch=%d.txt" % Max_epoch

            confusion_test = np.loadtxt(filepath_confusion_test)
            confusion_train = np.loadtxt(filepath_confusion_train)

            filename_confusion_test_save = "confusion_matrix_test_sample%d.txt" % sample
            filename_confusion_train_save = "confusion_matrix_training_sample%d.txt" % sample
            np.savetxt(save_dir_data +
                       filename_confusion_train_save, confusion_train)
            np.savetxt(save_dir_data +
                       filename_confusion_test_save, confusion_test)

            confusion_test_total = np.sum(confusion_test, axis=1)
            confusion_train_total = np.sum(confusion_train, axis=1)

            filename_confusion_test_save_total = "confusion_matrix_test_sample%d_total.txt" % sample
            filename_confusion_train_save_total = "confusion_matrix_training_sample%d_total.txt" % sample
            np.savetxt(
                save_dir_data + filename_confusion_train_save_total, confusion_train_total)
            np.savetxt(save_dir_data +
                       filename_confusion_test_save_total, confusion_test_total)

    num_sample = len(listname_list)

    for model_type in name_list:

        save_dir = work_dir + \
            "/summary_%s/%s/%s/" % (version, data_type_name, model_type)
        save_dir_data = save_dir + "data/"

        confusion_test_all = np.zeros((num_sample, 3, 3))
        confusion_train_all = np.zeros((num_sample, 3, 3))

        for sample, listname in enumerate(listname_list):

            filename_confusion_test_save = "confusion_matrix_test_sample%d.txt" % sample
            filename_confusion_train_save = "confusion_matrix_training_sample%d.txt" % sample
            confusion_train = np.loadtxt(
                save_dir_data + filename_confusion_train_save)
            confusion_test = np.loadtxt(
                save_dir_data + filename_confusion_test_save)

            confusion_train_all[sample, :, :] = confusion_train
            confusion_test_all[sample, :, :] = confusion_test

        filename_confusion_train_save_all = "confusion_matrix_training_all.npy"
        filename_confusion_test_save_all = "confusion_matrix_test_all.npy"

        np.save(save_dir_data + filename_confusion_train_save_all,
                confusion_train_all)
        np.save(save_dir_data + filename_confusion_test_save_all,
                confusion_test_all)

        filename_confusion_train_save_all_average = "confusion_matrix_training_all_average.csv"
        filename_confusion_test_save_all_average = "confusion_matrix_test_all_average.csv"

        filename_confusion_train_save_all_std = "confusion_matrix_training_all_std.csv"
        filename_confusion_test_save_all_std = "confusion_matrix_test_all_std.csv"

        confusion_train_all_average = np.average(confusion_train_all, axis=0)
        confusion_test_all_average = np.average(confusion_test_all, axis=0)

        confusion_train_all_std = np.std(confusion_train_all, axis=0, ddof=1)
        confusion_test_all_std = np.std(confusion_test_all, axis=0, ddof=1)

        np.savetxt(save_dir_data + filename_confusion_train_save_all_average,
                   confusion_train_all_average)
        np.savetxt(save_dir_data + filename_confusion_test_save_all_average,
                   confusion_test_all_average)
        np.savetxt(save_dir_data + filename_confusion_train_save_all_std,
                   confusion_train_all_std)
        np.savetxt(save_dir_data + filename_confusion_test_save_all_std,
                   confusion_test_all_std)
filename_MaxF1_epoch = "/ROC_time/data/MaxMacroF1_epoch.pickle"
filename_Max_epoch = filename_MaxF1_epoch
data_type_name = "ConfusionMatrixForMaxF1Model"
AverageConfusionMatrixOverSamples(
    filename_Max_epoch, listname_list, name_list, data_type_name, version, work_dir)

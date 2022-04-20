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

import matplotlib.pyplot as plt
import numpy as np


def SingleDataPlot(dirpath_fig, fig_title, xlabel, ylabel, data_x, data_y, figx, figy, left, right, bottom, top, dpi):
    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    ax.plot(data_x, data_y)
    ax.set_title(fig_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    filename_fig_png = fig_title + ".png"
    plt.savefig(dirpath_fig + filename_fig_png,
                format="png", dpi=dpi, transparent=True)
    filename_fig_pdf = fig_title + ".pdf"
    plt.savefig(dirpath_fig + filename_fig_pdf, dpi=dpi, transparent=True)
    filename_total_num_cell_list_x = fig_title + "_x.npy"
    filename_total_num_cell_list_y = fig_title + "_y.npy"
    np.save(dirpath_fig + filename_total_num_cell_list_x, data_x)
    np.save(dirpath_fig + filename_total_num_cell_list_y, data_y)

    plt.show()


def MultipleDataPlot(dirpath_fig, fig_title, xlabel, ylabel, data_x_list, data_y_list, figx, figy, left, right, bottom, top, dpi):
    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for i, data_x in enumerate(data_x_list):
        data_y = data_y_list[i]
        ax.plot(data_x, data_y)

        filename_total_num_cell_list_x = fig_title + "_x%d.npy" % i
        filename_total_num_cell_list_y = fig_title + "_y%d.npy" % i
        np.save(dirpath_fig + filename_total_num_cell_list_x, data_x)
        np.save(dirpath_fig + filename_total_num_cell_list_y, data_y)

    ax.set_title(fig_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    filename_fig_png = fig_title + ".png"
    plt.savefig(dirpath_fig + filename_fig_png,
                format="png", dpi=dpi, transparent=True)
    filename_fig_pdf = fig_title + ".pdf"
    plt.savefig(dirpath_fig + filename_fig_pdf, dpi=dpi, transparent=True)

    plt.show()


# +
def make_cmlist2(cm_name, n_color, max_index):
    cm_load = plt.get_cmap(cm_name)

    cm_step = np.uint8(max_index/(n_color-1))

    colorlist = []
    for i in range(n_color):
        index = int(cm_step * i)
        colorlist.append(cm_load(index))

    return colorlist


def make_cmlist_to_dense(cm_name, n_color, max_index):
    cm_load = plt.get_cmap(cm_name)

    cm_step = np.uint8(max_index/(n_color-1))

    colorlist = []
    for i in range(n_color):
        index = int(cm_step * (n_color-1-i))
        colorlist.append(cm_load(index))

    return colorlist


# +
###### Plot Maximum AUC #####

""" Not used
def PlotAUCTimeEvolution(p_abs, cmap, figx, figy, left, right, bottom, top, dpi):

    base_dir = str(p_abs) + "/"

    parameter_path = base_dir + "parameters.pickle"

    param = sutil.PickleLoad(parameter_path)

    hid_node = param["hid_node"]
    feature = 'feature_concat'
    time_list = param["time_list"]
    p_hidden = param["p_hidden"]
    in_plane = param["in_plane"]
    NoSelfInfo = param["NoSelfInfo"]
    feature_self = "AllZero"
    n_layers = param["n_layers"]

    skip = param["skip"]
    input_size = param["input_size"]
    in_feats = input_size
    epoch = param["epoch_total"]

    network_name = param["network_num"]
    crop_width = param["crop_width"]
    crop_height = param["crop_height"]

    model_rate_save = param["model_save_rate"]

    data_path = base_dir + "data_path/base_path_list_test.txt"

    data_path_test = sutil.Read1LineText(data_path)
    data_path_test = data_path_test.replace('\n', '')

    dir_ROC_time_data = base_dir + "ROC_time/data/"
    sutil.MakeDirs(dir_ROC_time_data)

    dir_ROC_time_figs = base_dir + "ROC_time/figs/"
    sutil.MakeDirs(dir_ROC_time_figs)

    files = glob.glob(base_dir + "labels/test/*")
    # print(len(files))
    n_network = len(files)-1
    # print(n_network)

    epoch_list = list(range(0, epoch, model_rate_save))
    epoch_list.append(epoch-1)
    # print(epoch_list)

    ################## calculate ROC ############

    AUC_list = []
    AUC_0_list = []
    AUC_1_list = []
    AUC_2_list = []
    #epoch_target = 300

    dir_labels = base_dir + "labels/"

    dir_ROC = base_dir + "ROC/"

    dir_ROC_figs = dir_ROC + "figs/"

    dir_ROC_data = dir_ROC + "data/"

    for epoch_target in epoch_list:

        ############# Test ################

        correct_label_all = np.empty((0), int)
        max_prob_all = np.empty(0)
        max_label_all = np.empty(0, int)

        prob_all = np.empty((0, 3))

        for count in range(n_network):

            dir_test = dir_labels + "test/test_data%d/" % count

            dir_test_logits_all = dir_labels + "test/test_data%d/Logits_all/" % count

            filename = "test_logits_all_epoch=%d.txt" % epoch_target
            logits_eval_np = np.loadtxt(dir_test_logits_all + filename)
            filename = "cellID_target.txt"

            cellID_np = np.loadtxt(dir_test + filename, dtype=int)

            # correct labels for the target cells
            dir_train_correct = dir_labels + "test/test_data%d/correct/" % count
            filename = "test_CorrLabel_target_epoch=%d.txt" % epoch_target
            labels_target_np = np.loadtxt(
                dir_train_correct + filename, dtype="int8")

            logits_eval_target = logits_eval_np[cellID_np]

            # print(logits_eval_target.shape)

            # print(logits_eval_target)

            prob = softmax(logits_eval_target, axis=1)

            # print(prob.shape)
            # print(m)

            max_prob = np.max(prob, axis=1)
            max_label = np.argmax(prob, axis=1)
            # print(max_label)

            correct_label_all = np.append(
                correct_label_all, labels_target_np, axis=0)
            max_prob_all = np.append(max_prob_all, max_prob, axis=0)
            max_label_all = np.append(max_label_all, max_label, axis=0)

            prob_all = np.append(prob_all, prob, axis=0)

        # print(correct_label_all.shape)
        # print(max_prob_all.shape)
        # print(max_label_all.shape)

        # print(prob_all.shape)

        # From https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        lw = 2

        y_score = prob_all

        # Binarize the output   # y_test is correct label
        y_test = label_binarize(correct_label_all, classes=[0, 1, 2])
        n_classes = y_test.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # print(fpr[1].shape)
        # print(tpr[1].shape)

        # Compute micro-average ROC curve and ROC area
        # print(y_test)

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves

        fig = plt.figure(figsize=(figx*2, figy*2))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

        # plt.figure()
        ax.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
                color='deeppink', linestyle='--', linewidth=lw)
        ax.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
                color='navy', linestyle='--', linewidth=lw)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC to multi-class')
        plt.legend(loc="lower right", fontsize=10)

        figname = dir_ROC_figs + "ROC_%d.pdf" % epoch_target
        plt.savefig(figname)
        figname_png = dir_ROC_figs + "ROC_%d.png" % epoch_target
        plt.savefig(figname_png, dpi=dpi)

        # plt.figure(figsize=(6,6))
        # plt.show()

        sutil.PickleDump(fpr, dir_ROC_data + "fpr_%d.pickle" % epoch_target)
        sutil.PickleDump(tpr, dir_ROC_data + "tpr_%d.pickle" % epoch_target)
        sutil.PickleDump(roc_auc, dir_ROC_data +
                         "roc_auc_%d.pickle" % epoch_target)

        AUC_list.append(roc_auc)

        AUC_0_list.append(roc_auc[0])
        AUC_1_list.append(roc_auc[1])
        AUC_2_list.append(roc_auc[2])

    sutil.PickleDump(AUC_list, dir_ROC_time_data + "AUC_time.pickle")

    AUC_list_list = [AUC_0_list, AUC_1_list, AUC_2_list]
    AUC_name_list = ["NB", "Dif", "Div"]

    mean = (np.array(AUC_0_list) + np.array(AUC_1_list) + np.array(AUC_2_list))/3
    sutil.PickleDump(AUC_list_list, dir_ROC_time_data + "AUC_all.pickle")
    sutil.PickleDump(AUC_name_list, dir_ROC_time_data + "AUC_name_all.pickle")

    sutil.PickleDump(mean, dir_ROC_time_data + "mean.pickle")

    max_arg = np.argmax(mean)
    max_value_list = np.array(
        [AUC_0_list[max_arg], AUC_1_list[max_arg], AUC_2_list[max_arg]])

    # print(epoch_list[max_arg])
    # print(max_value_list)

    sutil.PickleDump(max_value_list, dir_ROC_time_data + "Max_AUC.pickle")
    sutil.PickleDump(epoch_list[max_arg],
                     dir_ROC_time_data + "Max_AUC_epoch.pickle")
    sutil.PickleDump(max_arg, dir_ROC_time_data + "Max_AUC_index.pickle")

    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for i, auc in enumerate(AUC_list_list):

        ax.plot(epoch_list, auc, label=AUC_name_list[i], c=cmap[i], lw=1)

    ax.plot(epoch_list, mean, label="Mean", linestyle="dashed", c="b", lw=1)
    ax.axhline(0.5, ls="--", color="k", lw=0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=6)

    figname = dir_ROC_time_figs + "AUC_alll.pdf"
    plt.savefig(figname)
    figname_png = dir_ROC_time_figs + "AUC_all.png"
    plt.savefig(figname_png, dpi=dpi, transparent=True)

    plt.show()
"""

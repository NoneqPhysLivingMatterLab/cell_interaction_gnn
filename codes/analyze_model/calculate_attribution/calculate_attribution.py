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

"""Calculate attribution"""

import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import calculate_attribution_functions as att_func
from training import gnn_models
from functions import system_utility as sutil
import os
import sys
version = 26
print("version=%d" % version)




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


plt.rcParams['font.family'] = yaml_fig_obj["font.family"]
plt.rcParams['xtick.direction'] = yaml_fig_obj["xtick.direction"]
plt.rcParams['ytick.direction'] = yaml_fig_obj["ytick.direction"]
plt.rcParams['xtick.major.width'] = yaml_fig_obj["xtick.major.width"]
plt.rcParams['ytick.major.width'] = yaml_fig_obj["ytick.major.width"]
plt.rcParams['font.size'] = yaml_fig_obj["font.size"]
plt.rcParams['axes.linewidth'] = yaml_fig_obj["axes.linewidth"]


# %%
# parameters for plot
filename_yml = 'input_calculate_attribution.yml'
path_yml = "./" + filename_yml
yaml_obj = sutil.LoadYml(path_yml)

n = yaml_obj["n"]

plot_option = yaml_obj["plot_option"]
# "correct" for only correct cells, all for all the cells
cell_select = yaml_obj["cell_select"]

ModelType = yaml_obj["ModelType"]

subnetwork_name = yaml_obj["subnetwork_name"]

ClassList = yaml_obj["ClassList"]


# %%
path_gpu = "./gpu.txt"

if os.path.isfile(path_gpu) == True:

    gpu_load = sutil.Read1LineText(path_gpu)
    gpu = gpu_load[0]
    gpu = int(gpu)

else:
    gpu = 0  # -1 for cpu , gpu = 0 or 1

if gpu != -1:  # if gpu==-1, use cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu

#print(th.cuda.device_count(), "GPUs available")
print(th.__version__)  # 0.4.0


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)
print(th.cuda.current_device())


# %%
print(version)

#ModelType_filename = "ModelType.txt"
#ModelType =sutil.Read1LineText(ModelType_filename)
print(ModelType)

if ModelType == "MaxACC":
    ModelType2 = "Max_ACC"
elif ModelType == "MaxAUC":
    ModelType2 = "Max_AUC"
else:
    ModelType2 = ModelType  # If the epoch is specified


basedir_filename = "base_path.txt"
base_dir = sutil.ReadLinesText(basedir_filename)
base_dir = sutil.RemoveCyberduckPrefixSuffix(base_dir)[0] + "/"
# print(base_dir)


# print(subnetwork_name)


parameter_path = base_dir + "parameters.pickle"

param = sutil.PickleLoad(parameter_path)

hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature, reg, norm_final = gnn_models.LoadTrainingParameters(
    param)


# %%


dir_ROC_time_data = base_dir + "ROC_time/data/"


if ModelType2.isdigit() == False:
    epoch_target = sutil.PickleLoad(
        dir_ROC_time_data + "%s_epoch.pickle" % ModelType2)  # For Max ACC
else:
    epoch_target = int(ModelType2)


model_path = base_dir + "model/model_epoch=%012d.pt" % (epoch_target)


# %%


num_time = len(time_list)

data_path = base_dir + "data_path/base_path_list_test.txt"

data_path_test_list = sutil.ReadLinesText(data_path)


IDlist_path_list = []

frame_start_list = []
frame_end_list = []


crop_width_test_load_list = []
crop_height_test_load_list = []


for j, data_path_test in enumerate(data_path_test_list):

    crop_width = crop_width_list_test[j]
    crop_height = crop_height_list_test[j]

    IDlist_path = data_path_test + \
        "analysis/network%s_crop_w=%d_h=%d/" % (
            subnetwork_name, crop_width, crop_height)

    networkdir_name = IDlist_path + \
        "network%s_num_w=%d_h=%d_time=%d/" % (
            subnetwork_name, crop_width, crop_height, num_time)

    files = os.listdir(networkdir_name)
    # print(files)

    files_in_train = [s for s in files if 'NetworkWithFeartures' in s]

    for i, file in enumerate(files_in_train):
        IDlist_path_list.append(IDlist_path)

        crop_width_test_load_list.append(crop_width)
        crop_height_test_load_list.append(crop_height)

        frame_start = i
        frame_end = i+num_time-1
        frame_start_list.append(frame_start)
        frame_end_list.append(frame_end)


# print(IDlist_path_list)
# print(frame_start_list)
# print(frame_end_list)
# print(crop_width_test_load_list)
# print(crop_height_test_load_list)


att_test_dir = base_dir + "attribution_n=%d_%s/test/" % (n, ModelType)

sutil.MakeDirs(att_test_dir)


# %%
if architecture == "SP":

    model = gnn_models.CellFateNet(input_size, in_feats, hid_node, 3, feature, time_list, p_hidden, in_plane, NoSelfInfo,
                                   feature_self, n_layers, skip, average_switch, edge_switch, input_size_edge, feature_edge_concat, reg, norm_final)
    model_ver = model.version


if architecture == "NSP":
    model = gnn_models.CellFateNetTimeReversal(input_size, in_feats, hid_node, 3, feature, time_list, p_hidden, in_plane,
                                               NoSelfInfo, feature_self, n_layers, skip, edge_switch, input_size_edge, feature_edge_concat, average_switch)
    model_ver = model.version


# %%
model = model.to(device)
model.load_state_dict(th.load(model_path))
model.eval()


num_time = len(time_list)
# network used in learning
networkdir_path = base_dir + "network/"
files = os.listdir(networkdir_path)
# print(files)

files_test_wrong = [s for s in files if 'test' in s]
# print(files_test_wrong)

files_test = []
for i, wrong_order in enumerate(files_test_wrong):
    tmp = "test_%d.pickle" % i
    files_test.append(tmp)

# print(files_test)


n_net = len(files_test)

if n_net != len(IDlist_path_list):
    print("# of net error")


# print(IDlist_path_list)
# print(frame_start_list)
# print(frame_end_list)

for i, IDlist_path in enumerate(IDlist_path_list):

    if __name__ == '__main__':
        time_start = time.time()

    crop_width = crop_width_test_load_list[i]
    crop_height = crop_height_test_load_list[i]

    save_dir = att_test_dir + "test_%d/" % i
    sutil.MakeDirs(save_dir)

    label0_dir = save_dir + "0/"
    label1_dir = save_dir + "1/"
    label2_dir = save_dir + "2/"
    sutil.MakeDirs(label0_dir)
    sutil.MakeDirs(label1_dir)
    sutil.MakeDirs(label2_dir)

    label_all_dir = save_dir + "all/"
    sutil.MakeDirs(label_all_dir)

    cellID_target_path = base_dir + "labels/test/test_data%d/cellID_target.txt" % i
    correct_label_path = base_dir + \
        "labels/test/test_data%d/correct/test_CorrLabel_target_epoch=%d.txt" % (
            i, epoch_target)
    predicted_label_path = base_dir + \
        "labels/test/test_data%d/prediction/test_PredLabel_target_epoch=%d.txt" % (
            i, epoch_target)

    cellID_target = np.loadtxt(cellID_target_path).astype(np.int64)
    correct_label = np.loadtxt(correct_label_path).astype(np.int64)
    predicted_label = np.loadtxt(predicted_label_path).astype(np.int64)

    frame_start = frame_start_list[i]
    frame_end = frame_end_list[i]

    network_path = networkdir_path + files_test[i]

    # print(network_path)
    # print(IDlist_path)
    # print("start:%d,end:%d"%(frame_start,frame_end))

    network_load = sutil.PickleLoad(network_path)

    #print("Original network")
    # print(network_load)

    input_template = network_load.to(device)
    input_original = copy.deepcopy(network_load)
    input_blank = copy.deepcopy(network_load)

    input_template_cp = copy.deepcopy(input_template)

    for time_name in time_list:
        input_blank.nodes[time_name].data[feature] = input_original.nodes[time_name].data[feature] * 0

    if edge_switch == 1:

        for time_name in time_list:
            interaction_tuple = (time_name, 'interaction', time_name)
            input_blank.edges[interaction_tuple].data[feature_edge_concat] = input_original.edges[interaction_tuple].data[feature_edge_concat] * 0

    nodeID_list = cellID_target
    total_node_number = len(nodeID_list)

    #y_score_list_all,IG_list_all = compute_integrated_gradient_AllNode(n,input_original,input_template,input_blank, model,time_list,feature,nodeID_list,ClassList)

    # Remove isolated IDs
    input_template_exist, existing_IDList, new_index = att_func.RemoveIsolatedNodesFromGraph(
        input_template_cp, nodeID_list, time_list)

    #print("New network")
    # print(input_template_exist)
    # print(existing_IDList)
    # print(new_index)

    input_original_exist = copy.deepcopy(input_template_exist)
    input_blank_exist = copy.deepcopy(input_template_exist)

    for time_name in time_list:
        input_blank_exist.nodes[time_name].data[feature] = input_original_exist.nodes[time_name].data[feature] * 0
    if edge_switch == 1:

        for time_name in time_list:
            interaction_tuple = (time_name, 'interaction', time_name)
            input_blank_exist.edges[interaction_tuple].data[feature_edge_concat] = input_original_exist.edges[interaction_tuple].data[feature_edge_concat] * 0

    #y_score_list_all,IG_list_all = compute_integrated_gradient_AllNode(n,input_original,input_template,input_blank, model,time_list,feature,nodeID_list,ClassList)

    y_score_list_all, IG_list_all = att_func.compute_integrated_gradient_AllNode2(
        n, existing_IDList, input_original_exist, input_template_exist, input_blank_exist, input_original, model, time_list, feature, new_index, nodeID_list, ClassList)

    sutil.PickleDump(y_score_list_all, label_all_dir +
                     "y_score_list_all.pickle")
    sutil.PickleDump(IG_list_all, label_all_dir + "IG_list_all.pickle")
    sutil.PickleDump(nodeID_list, label_all_dir + "nodeID_list.pickle")

    for node_count, nodeID in enumerate(nodeID_list):

        # This is predicted label, but
        PredictedClass = predicted_label[node_count]
        # This is predicted label, but
        CorrectClass = correct_label[node_count]

        # Index for the IG of the correct label of the target cell
        index = CorrectClass*total_node_number + node_count

        y_score_list = y_score_list_all[node_count, CorrectClass, :]
        IntegratedGradient_list = IG_list_all[index]

        # correct_label[node_count]
        #print("nodeID%d, Predictedlabel%d, Correctlabel%d"%(nodeID,PredictedClass,CorrectClass))

        if predicted_label[node_count] == correct_label[node_count]:
            cellIDdir_save = save_dir + \
                "%d/cellID=%d_correct/" % (CorrectClass, nodeID)
        else:
            cellIDdir_save = save_dir + \
                "%d/cellID=%d_wrong/" % (CorrectClass, nodeID)
        sutil.MakeDirs(cellIDdir_save)

        AllTargetCellListNoDoubling_path = IDlist_path + "network%s_%s_AllTargetCellListNoDoubling_noborder_num_w=%d_h=%d_time=%d/t=%dto%d/AllTargetCellListNoDoubling_cellID=%d_t=%dto%d.pickle" % (
            subnetwork_name, architecture, crop_width, crop_height, num_time, frame_start, frame_end, nodeID, frame_start, frame_end)
        AllTargetCellListNoDoubling = sutil.PickleLoad(
            AllTargetCellListNoDoubling_path)

        AllNeighborCellList_path = IDlist_path + "network%s_%s_AllNeighborCellList_noborder_num_w=%d_h=%d_time=%d/t=%dto%d/AllNeighborCellList_cellID=%d_t=%dto%d.pickle" % (
            subnetwork_name, architecture, crop_width, crop_height, num_time, frame_start, frame_end, nodeID, frame_start, frame_end)
        AllNeighborCellList = sutil.PickleLoad(AllNeighborCellList_path)

        LineageList_path = IDlist_path + "network%s_%s_LineageList_noborder_num_w=%d_h=%d_time=%d/t=%dto%d/LineageList_cellID=%d_t=%dto%d.pickle" % (
            subnetwork_name, architecture, crop_width, crop_height, num_time, frame_start, frame_end, nodeID, frame_start, frame_end)
        LineageList = sutil.PickleLoad(LineageList_path)

        y_score_list_save = cellIDdir_save + "y_score_list.pickle"
        IntegratedGradient_list_save = cellIDdir_save + "IntegratedGradient_list.pickle"
        AllTargetCellListNoDoubling_save = cellIDdir_save + \
            "AllTargetCellListNoDoubling.pickle"
        AllNeighborCellList_save = cellIDdir_save + "AllNeighborCellList.pickle"
        LineageList_save = cellIDdir_save + "LineageList.pickle"

        sutil.PickleDump(y_score_list, y_score_list_save)
        sutil.PickleDump(IntegratedGradient_list, IntegratedGradient_list_save)
        sutil.PickleDump(AllTargetCellListNoDoubling,
                         AllTargetCellListNoDoubling_save)
        sutil.PickleDump(AllNeighborCellList, AllNeighborCellList_save)
        sutil.PickleDump(LineageList, LineageList_save)

        if plot_option == 1:
            plt.plot(np.array(range(n+1))/n, y_score_list)
            # plt.show()
            title = "CellID=%d_label=%d" % (nodeID, CorrectClass)
            plt.title(title)
            plt.xlabel('alpha')
            plt.ylabel('p')

            plt.savefig(cellIDdir_save + title + ".png", dpi=dpi)
            plt.close()

    elapsed_time = time.time()-time_start
    time_rec = "elapsed_time:{0}".format(elapsed_time) + "[sec]"
    print(time_rec)
    time_filename = att_test_dir + "/time_network%d.txt" % i
    file_time = open(time_filename, "w")
    file_time.write(time_rec)
    file_time.close()


# %%
# Attribution all
# Originally AllCellAttributionVer4.py

# network used in learning
networkdir_path = base_dir + "network/"
files = os.listdir(networkdir_path)

files_test_wrong = [s for s in files if 'test' in s]
# print(files_test_wrong)

files_test = []
for i, wrong_order in enumerate(files_test_wrong):
    tmp = "test_%d.pickle" % i
    files_test.append(tmp)

# print(files_test)


#files_train = [s for s in files if 'train' in s]
# print(files_train)

n_net = len(files_test)

if n_net != len(IDlist_path_list):
    print("# of net error")


# for i in range(n_net):

# print(IDlist_path_list)
# print(frame_start_list)
# print(frame_end_list)

for i, IDlist_path in enumerate(IDlist_path_list):

    if __name__ == '__main__':
        time_start = time.time()

    crop_width = crop_width_test_load_list[i]
    crop_height = crop_height_test_load_list[i]

    save_dir = att_test_dir + "test_%d/" % i

    label0_dir = save_dir + "AllCells/0/"
    label1_dir = save_dir + "AllCells/1/"
    label2_dir = save_dir + "AllCells/2/"

    sutil.MakeDirs(label0_dir)
    sutil.MakeDirs(label1_dir)
    sutil.MakeDirs(label2_dir)

    label_all_dir = save_dir + "all/"

    cellID_target_path = base_dir + "labels/test/test_data%d/cellID_target.txt" % i
    correct_label_path = base_dir + \
        "labels/test/test_data%d/correct/test_CorrLabel_target_epoch=%d.txt" % (
            i, epoch_target)
    predicted_label_path = base_dir + \
        "labels/test/test_data%d/prediction/test_PredLabel_target_epoch=%d.txt" % (
            i, epoch_target)

    cellID_target = np.loadtxt(cellID_target_path).astype(np.int64)
    correct_label = np.loadtxt(correct_label_path).astype(np.int64)
    predicted_label = np.loadtxt(predicted_label_path).astype(np.int64)

    frame_start = frame_start_list[i]
    frame_end = frame_end_list[i]

    network_path = networkdir_path + files_test[i]

    # print(network_path)
    # print(IDlist_path)
    # print("start:%d,end:%d"%(frame_start,frame_end))

    nodeID_list = cellID_target
    total_node_number = len(nodeID_list)

    y_score_list_all = sutil.PickleLoad(
        label_all_dir + "y_score_list_all.pickle")

    IG_list_all = sutil.PickleLoad(label_all_dir + "IG_list_all.pickle")

    nodeID_list = sutil.PickleLoad(label_all_dir + "nodeID_list.pickle")

    for node_count, nodeID in enumerate(nodeID_list):

        # This is predicted label, but
        PredictedClass = predicted_label[node_count]

        for Class in ClassList:  # For all labels
            # Index for the IG of the correct label of the target cell
            index = Class*total_node_number + node_count

            y_score_list = y_score_list_all[node_count, Class, :]
            IntegratedGradient_list = IG_list_all[index]

            cellIDdir_save = save_dir + \
                "AllCells/%d/cellID=%d_reallabel=%d/" % (
                    Class, nodeID, correct_label[node_count])
            sutil.MakeDirs(cellIDdir_save)

            AllTargetCellListNoDoubling_path = IDlist_path + "network%s_%s_AllTargetCellListNoDoubling_noborder_num_w=%d_h=%d_time=%d/t=%dto%d/AllTargetCellListNoDoubling_cellID=%d_t=%dto%d.pickle" % (
                subnetwork_name, architecture, crop_width, crop_height, num_time, frame_start, frame_end, nodeID, frame_start, frame_end)
            AllTargetCellListNoDoubling = sutil.PickleLoad(
                AllTargetCellListNoDoubling_path)

            AllNeighborCellList_path = IDlist_path + "network%s_%s_AllNeighborCellList_noborder_num_w=%d_h=%d_time=%d/t=%dto%d/AllNeighborCellList_cellID=%d_t=%dto%d.pickle" % (
                subnetwork_name, architecture, crop_width, crop_height, num_time, frame_start, frame_end, nodeID, frame_start, frame_end)
            AllNeighborCellList = sutil.PickleLoad(AllNeighborCellList_path)

            LineageList_path = IDlist_path + "network%s_%s_LineageList_noborder_num_w=%d_h=%d_time=%d/t=%dto%d/LineageList_cellID=%d_t=%dto%d.pickle" % (
                subnetwork_name, architecture, crop_width, crop_height, num_time, frame_start, frame_end, nodeID, frame_start, frame_end)
            LineageList = sutil.PickleLoad(LineageList_path)

            y_score_list_save = cellIDdir_save + "y_score_list.pickle"
            IntegratedGradient_list_save = cellIDdir_save + "IntegratedGradient_list.pickle"
            AllTargetCellListNoDoubling_save = cellIDdir_save + \
                "AllTargetCellListNoDoubling.pickle"
            AllNeighborCellList_save = cellIDdir_save + "AllNeighborCellList.pickle"
            LineageList_save = cellIDdir_save + "LineageList.pickle"

            sutil.PickleDump(y_score_list, y_score_list_save)
            sutil.PickleDump(IntegratedGradient_list,
                             IntegratedGradient_list_save)
            sutil.PickleDump(AllTargetCellListNoDoubling,
                             AllTargetCellListNoDoubling_save)
            sutil.PickleDump(AllNeighborCellList, AllNeighborCellList_save)
            sutil.PickleDump(LineageList, LineageList_save)

            if plot_option == 1:
                plt.plot(np.array(range(n+1))/n, y_score_list)
                # plt.show()
                title = "CellID=%d_label=%d" % (nodeID, CorrectClass)
                plt.title(title)
                plt.xlabel('alpha')
                plt.ylabel('p')

                plt.savefig(cellIDdir_save + title + ".png", dpi=dpi)
                plt.close()

    elapsed_time = time.time()-time_start
    time_rec = "elapsed_time:{0}".format(elapsed_time) + "[sec]"
    # print(time_rec)
    time_filename = att_test_dir + "/time_network%d.txt" % i
    file_time = open(time_filename, "w")
    file_time.write(time_rec)
    file_time.close()


# %%

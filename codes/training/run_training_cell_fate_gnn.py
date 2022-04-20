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

"""Training GNN to predict cell fate"""

from functions import system_utility as sutil
from sklearn.metrics import balanced_accuracy_score
from functions import plot_module as plot_module
import gnn_models
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch as th
import os
import sys
ver = 2



# +
# parameters for plot
filename_fig_yml = 'input_fig_config.yml'
path_fig_yml = "./" + filename_fig_yml
yaml_fig_obj = sutil.LoadYml(path_fig_yml)

plt.rcParams['font.family'] = yaml_fig_obj["font.family"]
plt.rcParams['xtick.direction'] = yaml_fig_obj["xtick.direction"]
plt.rcParams['ytick.direction'] = yaml_fig_obj["ytick.direction"]
plt.rcParams['xtick.major.width'] = yaml_fig_obj["xtick.major.width"]
plt.rcParams['ytick.major.width'] = yaml_fig_obj["ytick.major.width"]
plt.rcParams['font.size'] = yaml_fig_obj["font.size"]
plt.rcParams['axes.linewidth'] = yaml_fig_obj["axes.linewidth"]

figx = yaml_fig_obj["figx"]
figy = yaml_fig_obj["figy"]
left = yaml_fig_obj["left"]
right = yaml_fig_obj["right"]
bottom = yaml_fig_obj["bottom"]
top = yaml_fig_obj["top"]
dpi = yaml_fig_obj["dpi"]


# +
# Load parameters
filename_yml = 'input_run_training_cell_fate_gnn.yml'
path_yml = "./" + filename_yml
yaml_obj = sutil.LoadYml(path_yml)


# Paths
dirpath_work = yaml_obj["dirpath_work"]
sutil.MakeDirs(dirpath_work)

base_path_list_training = yaml_obj["base_path_list_training"]
base_path_list_test = yaml_obj["base_path_list_test"]

# Convert relative path to absolute path
base_path_list_training = [
    str(pathlib.Path(i).resolve()) + "/" for i in base_path_list_training]
base_path_list_test = [str(pathlib.Path(i).resolve()) +
                       "/" for i in base_path_list_test]


# Input parameters
step_total = yaml_obj["step_total"]  # number of epoch
# rate of display of results (unit:epoch)
display_rate = yaml_obj["display_rate"]
# rate of saving models (unit:epoch)
model_save_rate = yaml_obj["model_save_rate"]

gpu = yaml_obj["gpu"]  # GPU number
# print(gpu)


# If 1, we introduce null graphs to train the model to satisfay the valid baselines for attribution. if 0, we don't introduce null graphs.
null_net = yaml_obj["null_net"]

# if average 1, mean aggregation for spatial message passing. If 0, sum aggregation.
average_switch = yaml_obj["average_switch"]

# SP: for unidirectional GNN model, NSP for bidirectional GNN model
architecture = yaml_obj["architecture"]


n_layers = yaml_obj["n_layers"]  # number of layers of a MLP
hid_node = yaml_obj["hid_node"]  # number of nodes in the hidden layer


# if 0, full model. if 1, cell external model.
NoSelfInfo = yaml_obj["NoSelfInfo"]

in_plane = yaml_obj["in_plane"]  # number of in-plane message passing


p_hidden = yaml_obj["p_hidden"]  # dropout rate


num_time = yaml_obj["num_time"]  # number of time frames. N_t.


# Select feature set used for the training.
# "Area_norm": normalized area, "G1Signal_norm":normalized G1 signal,
# "celltype_future_onehot2": future fate. this feature set to null for the final frame of each network.
# "random_feature": uniform random numbers [0,1]
# "zero": null vector
feature_list = yaml_obj["feature_list"]

opt_name = yaml_obj["opt_name"]  # name of optimizer. Use adam.
learning_rate = yaml_obj["learning_rate"]    # learning rate

n_label = yaml_obj["n_label"]    # learning rate

fate_label = yaml_obj["fate_label"]

# +
# Make dirs to output results
dirname_result = "result"
dirpath_result = dirpath_work + dirname_result + "/"

# print(dirpath_result)
sutil.MakeDirs(dirpath_result)

path_gpu = dirpath_work + "gpu.txt"

# +
# Load advanced parameters
# Other parameters which are not necessary to change in this demo program. Please don't change.


# if 0, softmax score for null graphs are even 1/3. if 1, it is set to the number ratio.
null_weight = yaml_obj["null_weight"]
# if num_even==1, the label number is set to equal for training data.
num_even = yaml_obj["num_even"]
edge_switch = yaml_obj["edge_switch"]  # if 1, use edge_feature
border = yaml_obj["border"]  # if zero, we don't use border cells.
dropout = yaml_obj["dropout"]
weight_decay = yaml_obj["weight_decay"]
# if NoSelfInfo = 1, we asign this feature.
feature_self = yaml_obj["feature_self"]

# Regularization option
# 0: no regularization, 1: node-wise (layernorm), 2: batch-norm (一つのtime network内で平均),3: node-wise (layernorm) 出力手前に入れない, 4: batch-norm (一つのtime network内で平均)出力手前に入れない
reg = yaml_obj["reg"]
# if zero, no norm in the final layer of each function
norm_final = yaml_obj["norm_final"]
cuda_clear = yaml_obj["cuda_clear"]


shuffle = yaml_obj["shuffle"]  # shuffle labels if 1. none if 0
network_num = yaml_obj["network_num"]
skip = yaml_obj["skip"]  # If 1, skip connection. If 0, no skip connection

partial_test = yaml_obj["partial_test"]  # if zero, use all the data


crop_width_list_train = yaml_obj["crop_width_list_train"]
crop_height_list_train = yaml_obj["crop_height_list_train"]
crop_width_list_test = yaml_obj["crop_width_list_test"]
crop_height_list_test = yaml_obj["crop_height_list_test"]


#############################

penalty_zero = yaml_obj["penalty_zero"]
iter_n = yaml_obj["iter_n"]
feature = yaml_obj["feature"]
feature_edge_concat = yaml_obj["feature_edge_concat"]


# Select Edge features
feature_list_edge = yaml_obj["feature_list_edge"]

omit_t0 = yaml_obj["omit_t0"]  # if 1, we omit the first frame


# +
# Define other parameters
if in_plane == 0:
    NoSelfInfo = 0

if os.path.isfile(path_gpu) == True:

    gpu_load = sutil.Read1LineText(path_gpu)
    gpu = gpu_load[0]
    gpu = int(gpu)

else:
    gpu = gpu  # -1 for cpu , gpu = 0 or 1


# Define time frame name list [t0,t1,t2,,
time_list = []

for i in range(num_time):
    time_list.append("t"+str(i))

final_time = time_list[-1]


# Define size of input features
input_size = len(feature_list)
input_size_edge = len(feature_list_edge)

if ('celltype_future_onehot' in feature_list) == True:
    input_size += 3
if ('celltype_future_onehot2' in feature_list) == True:
    input_size += 2

if ('zero_celltype_future_onehot2' in feature_list) == True:
    input_size += 2

if ('lineage_onehot' in feature_list) == True:
    input_size += 2

if ('lineage_onehot2' in feature_list) == True:
    input_size += 1

if ('zero_lineage_onehot2' in feature_list) == True:
    input_size += 1

if ('SibFate_onehot' in feature_list) == True:
    input_size += 3

if ('zero_SibFate_onehot' in feature_list) == True:
    input_size += 3

if ('O_celltype_future_onehot' in feature_list) == True:
    input_size += 3

if ('O_lineage_onehot' in feature_list) == True:
    input_size += 2

if ('O_SibFate_onehot' in feature_list) == True:
    input_size += 3


in_feats = input_size


# Input dammy crop_width to keep the shape of the parameter file.
crop_width = 0  # for training
crop_height = crop_width  # for training
crop_width_test = 0
crop_height_test = crop_width_test


epoch_total = int(step_total/iter_n)


# +
# Create list of parameters. Please don't change.
# TODO: Replace this with dictionary
parameters_set = [gpu, shuffle, network_num, model_save_rate, n_layers, partial_test, NoSelfInfo, in_plane, border, dropout, p_hidden, weight_decay, time_list, crop_width, crop_width_test,
                  penalty_zero, hid_node, step_total, iter_n, feature_list, omit_t0, learning_rate, opt_name, display_rate, input_size, in_feats, skip, step_total, null_net, average_switch, architecture,
                  feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear,
                  reg, norm_final, null_weight, num_even]


parameters_set = {"gpu": gpu,
                  "shuffle": shuffle,
                  "network_num": network_num,
                  "model_save_rate": model_save_rate,
                  "n_layers": n_layers,
                  "partial_test": partial_test,
                  "NoSelfInfo": NoSelfInfo,
                  "in_plane": in_plane,
                  "border": border,
                  "dropout": dropout,
                  "p_hidden": p_hidden,
                  "weight_decay": weight_decay,
                  "time_list": time_list,
                  "crop_width": crop_width,
                  "crop_width_test": crop_width_test,
                  "penalty_zero": penalty_zero,
                  "hid_node": hid_node,
                  "step_total": step_total,
                  "iter_n": iter_n,
                  "feature_list": feature_list,
                  "omit_t0": omit_t0,
                  "learning_rate": learning_rate,
                  "opt_name": opt_name,
                  "display_rate": display_rate,
                  "input_size": input_size,
                  "in_feats": in_feats,
                  "skip": skip,
                  "step_total": step_total,
                  "null_net": null_net,
                  "average_switch": average_switch,
                  "architecture": architecture,
                  "feature_list_edge": feature_list_edge,
                  "feature_edge_concat": feature_edge_concat,
                  "input_size_edge": input_size_edge,
                  "edge_switch": edge_switch,
                  "crop_width_list_train": crop_width_list_train,
                  "crop_height_list_train": crop_height_list_train,
                  "crop_width_list_test": crop_width_list_test,
                  "crop_height_list_test": crop_height_list_test,
                  "cuda_clear": cuda_clear,
                  "reg": reg,
                  "norm_final": norm_final,
                  "null_weight": null_weight,
                  "num_even": num_even,
                  "epoch_total": epoch_total,
                  "feature": feature,
                  "n_label": n_label,
                  "fate_label": fate_label,
                  "base_path_list_training": base_path_list_training,
                  "base_path_list_test": base_path_list_test,
                  "final_time": final_time,
                  "crop_width": crop_width,
                  "crop_height": crop_height,
                  "crop_width_test": crop_width_test,
                  "crop_height_test": crop_height_test}


# +

training_path_list, cellID_training_path_list = gnn_models.ReturnListFilenamesNetworkAndID(base_path_list_training,
                                                                                           crop_width_list_train, crop_height_list_train,
                                                                                           network_num, num_time, omit_t0, shuffle, border)

test_path_list, cellID_test_path_list = gnn_models.ReturnListFilenamesNetworkAndID(base_path_list_test, crop_width_list_test, crop_height_list_test,
                                                                                   network_num, num_time, omit_t0, shuffle, border)


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
# Calculate weight for cross-sensitive learning
# Weight is calculated from the training data set

num0, num1, num2, weight_cs, weight_c0, weight_c1, weight_c2 = gnn_models.CalculateWeightTrainingData(
    training_path_list, cellID_training_path_list, final_time, penalty_zero)


# +
# Define model, optimizer, Loss


if architecture == "SP":

    model = gnn_models.CellFateNet(input_size, in_feats, hid_node, 3, feature, time_list, p_hidden, in_plane, NoSelfInfo,
                                   feature_self, n_layers, skip, average_switch, edge_switch, input_size_edge, feature_edge_concat, reg, norm_final)
    model_ver = model.version


if architecture == "NSP":
    model = gnn_models.CellFateNetTimeReversal(input_size, in_feats, hid_node, 3, feature, time_list, p_hidden, in_plane,
                                               NoSelfInfo, feature_self, n_layers, skip, edge_switch, input_size_edge, feature_edge_concat, average_switch)
    model_ver = model.version


if gpu != -1:
    model = model.to(device)

if opt_name == "adam":
    opt = th.optim.Adam(model.parameters(), lr=learning_rate,
                        weight_decay=weight_decay)

if opt_name == "sgd":
    opt = th.optim.SGD(model.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)

if opt_name == "rmsprop":
    opt = th.optim.RMSprop(model.parameters(), weight_decay=weight_decay)

if gpu != -1:
    weight_cs = weight_cs.to(device)


criterion = nn.CrossEntropyLoss(weight=weight_cs)  # loss for original graphs
criterion2 = nn.MSELoss()  # loss for null graphs


# +
# Make result directories, define default arrays for outputs.


#learning_dir_name = "./result/"
learning_dir_path = dirpath_result

result_dir_name_base = "%s_fnum%d_time%d_net%s_ptest%d_noself%d_inplane%d_skip%d_nullnet%d_ave%d_edge%d_reg%d-%d_nullw%d_even%d/" % (
    architecture, input_size, num_time, network_num, partial_test, NoSelfInfo, in_plane, skip, null_net, average_switch, edge_switch, reg, norm_final, null_weight, num_even)
result_dir_name = result_dir_name_base + "model=%d_epoch=%d_hidnode=%d_lr=%f_opt=%s_feature=%s_infeats=%d_penalty=%.1f_shuffle=%d_noself=%d_drop=%.2f_wd=%.6f_inplane=%d_omitt0=%d_border=%d_ptest=%d_layer=%d" % (
    model_ver, epoch_total, hid_node, learning_rate, opt_name, feature, in_feats, penalty_zero, shuffle, NoSelfInfo, p_hidden, weight_decay, in_plane, omit_t0, border, partial_test, n_layers)

#result_dir_name = "sample"

result_dir_path = learning_dir_path + result_dir_name

for i in range(50):
    if not os.path.isdir(result_dir_path + "_%d/" % i):
        result_dir_path = result_dir_path + "_%d/" % i
        break

p_rel = pathlib.Path(result_dir_path)
p_abs = p_rel.resolve()
# print(p_abs)

dir_fig = result_dir_path + "fig/"
sutil.MakeDirs(dir_fig)

dir_log = result_dir_path + "log/"
sutil.MakeDirs(dir_log)

dir_model = result_dir_path + "model/"
sutil.MakeDirs(dir_model)

dir_labels = result_dir_path + "labels/"
sutil.MakeDirs(dir_labels)

dir_network = result_dir_path + "network/"
sutil.MakeDirs(dir_network)

dir_features = result_dir_path + "features/"
sutil.MakeDirs(dir_features)

dir_data_path = result_dir_path + "data_path/"
sutil.MakeDirs(dir_data_path)

sutil.PickleDump(parameters_set, result_dir_path + "parameters.pickle")


weight_cs = weight_cs.to("cpu").detach().numpy().copy()
np.savetxt(dir_log+'/weight_cs.txt', weight_cs)

sutil.SaveListText(base_path_list_training, dir_data_path +
                   "base_path_list_training.txt")
sutil.SaveListText(base_path_list_test, dir_data_path +
                   "base_path_list_test.txt")

sutil.SaveListText(training_path_list, dir_data_path +
                   "network_path_list_training.txt")
sutil.SaveListText(test_path_list, dir_data_path +
                   "network_path_list_test.txt")

sutil.SaveListText(cellID_training_path_list,
                   dir_data_path + "CellID_path_list_training.txt")
sutil.SaveListText(cellID_test_path_list, dir_data_path +
                   "CellID_path_list_test.txt")


if num_even == 1:

    cellID_0_all, cellID_1_all, cellID_2_all, cellID_list_all = gnn_models.MakeCellIDListWithEqualNumbers(
        num0, num1, num2, training_path_list, final_time, cellID_training_path_list)

    cellID_even_path = result_dir_path + "cellID_even/"
    sutil.MakeDirs(cellID_even_path)

    sutil.PickleDump(cellID_0_all, cellID_even_path +
                     "label0_cellID_even.pickle")
    sutil.PickleDump(cellID_1_all, cellID_even_path +
                     "label1_cellID_even.pickle")
    sutil.PickleDump(cellID_2_all, cellID_even_path +
                     "label2_cellID_even.pickle")
    sutil.PickleDump(cellID_list_all, cellID_even_path +
                     "AllLabel_cellID_even.pickle")


# +
# concatenate features in feature_list
# random feature is generated here.

filename = "features.txt"
feature_path = dir_features + filename

f = open(feature_path, 'w')
for x in feature_list:
    f.write(str(x) + "\n")
f.close()

filename_edge = "features_edge.txt"
feature_path_edge = dir_features + filename_edge

f = open(feature_path_edge, 'w')
for x in feature_list_edge:
    f.write(str(x) + "\n")
f.close()


train_data_load, test_data_load, train_data_null = gnn_models.ConcatenateFeatures(
    training_path_list, test_path_list, time_list, feature, feature_self, dir_network, feature_list, device, gpu, null_net, NoSelfInfo, edge_switch, feature_list_edge)


# -

def RunTrainingCellFateGNN(model, training_path_list, test_path_list, cellID_training_path_list, cellID_test_path_list, train_data_load, test_data_load, train_data_null, n_label, device, parameters_set, final_time, dir_model, dir_labels, dir_log):

    # Load paramters
    epoch_total = parameters_set["epoch_total"]
    gpu = parameters_set["gpu"]
    model_save_rate = parameters_set["model_save_rate"]

    num_even = parameters_set["num_even"]
    null_weight = parameters_set["null_weight"]

    iter_n = parameters_set["iter_n"]

    feature = parameters_set["feature"]

    # print(feature)

    # Initialize accuracy arrays
    n_train = len(training_path_list)
    history_train = np.zeros((epoch_total, n_train))

    if gpu != -1:
        history_train = th.from_numpy(history_train)
        history_train = history_train.to(device)

    n_eval = len(test_path_list)

    history_eval = np.zeros((epoch_total, n_eval))

    if gpu != -1:
        history_eval = th.from_numpy(history_eval)

        history_eval = history_eval.to(device)

    last_epoch = epoch_total

    accuracy_eval = np.zeros((epoch_total, n_eval))
    accuracy_train = np.zeros((epoch_total, n_train))

    accuracy_balance_eval = np.zeros(epoch_total)
    accuracy_balance_train = np.zeros(epoch_total)

    #######   TRAINING #######
    # Evaluation and then training for every epoch

    model.train()

    for epoch in range(epoch_total):

        ############# SAVE model################
        PATH = dir_model+'/model_epoch=%012d.pt' % epoch
        if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
            th.save(model.state_dict(), PATH)

        # Evaluation

        model.eval()  # we need this for Dropout
        with th.no_grad():

            ############# training loss & accuracy################

            acc_list_all = np.zeros((n_train, n_label, n_label))

            # labels for all the data (concatenated)
            labels_target_all_train = np.array([])
            prediction_target_all_train = np.array([])

            for count, g_path in enumerate(training_path_list):

                g = train_data_load[count]
                labels = g.nodes[final_time].data['celltype']

                cellID_file_path = cellID_training_path_list[count]
                cellID = np.loadtxt(cellID_file_path).astype(np.int64)

                if num_even == 1:
                    cellID = np.array(cellID_list_all[count]).astype(np.int64)

                if gpu != -1:
                    cellID = th.from_numpy(cellID).to(device)

                logits = model(g)

                loss = criterion(logits[cellID], labels[cellID])

                history_train[epoch, count] = loss.data

                prediction = logits.data.max(1)[1]

                labels_target = labels[cellID]
                prediction_target = prediction[cellID]

                if gpu != -1:
                    labels_target = labels_target.to(
                        "cpu")  # .detach().numpy().copy()
                    prediction_target = prediction_target.to(
                        "cpu")  # .detach().numpy().copy()

                labels_target_all_train = np.append(
                    labels_target_all_train, labels_target)
                prediction_target_all_train = np.append(
                    prediction_target_all_train, prediction_target)

                accuracy = prediction_target.eq(
                    labels_target).sum().numpy() / len(labels_target)  # 正解率

                accuracy_train[epoch, count] = accuracy

                dir_train = dir_labels + "training/training_data%d/" % count
                dir_train_predictioin = dir_labels + \
                    "training/training_data%d/prediction/" % count
                sutil.MakeDirs(dir_train_predictioin)
                filename = "training_PredLabel_target_epoch=%d.txt" % epoch
                prediction_target_np = prediction_target.to(
                    'cpu').detach().numpy().copy()

                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_train_predictioin +
                               filename, prediction_target_np)

                dir_train_correct = dir_labels + "training/training_data%d/correct/" % count
                sutil.MakeDirs(dir_train_correct)

                filename = "training_CorrLabel_target_epoch=%d.txt" % epoch
                labels_target_np = labels_target.to(
                    'cpu').detach().numpy().copy()

                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_train_correct + filename, labels_target_np)

                dir_train_logits_all = dir_labels + "training/training_data%d/Logits_all/" % count
                sutil.MakeDirs(dir_train_logits_all)

                filename = "training_logits_all_epoch=%d.txt" % epoch
                logits_np = logits.to('cpu').detach().numpy().copy()
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_train_logits_all + filename, logits_np)

                filename = "labels_all.txt"
                labels_np = labels.to('cpu').detach().numpy().copy()
                if epoch % model_save_rate == 0:
                    np.savetxt(dir_train + filename, labels_np, fmt="%d")

                filename = "cellID_target.txt"
                if gpu != -1:
                    cellID_np = cellID.to('cpu').detach().numpy().copy()
                    if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                        np.savetxt(dir_train + filename, cellID_np, fmt="%d")
                else:
                    if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                        np.savetxt(dir_train + filename, cellID, fmt="%d")

                # Make chart of  CorrectVsPredict

                acc_list = np.zeros((n_label, n_label))
                acc_list_ratio = np.zeros((n_label, n_label))
                for i in range(n_label):

                    pred_temp = prediction_target_np[np.where(
                        labels_target_np == i)]

                    len1 = len(pred_temp)

                    for j in range(n_label):

                        len2 = len(pred_temp[np.where(pred_temp == j)])

                        acc_list[i, j] = len2
                        if len1 != 0:
                            acc_list_ratio[i, j] = len2/len1

                acc_list_all[count, :, :] = acc_list

                dir_train_cvsp = dir_labels + "training/training_data%d/CorrectVsPredict/" % count
                sutil.MakeDirs(dir_train_cvsp)
                filename = "training_CorrectVsPredict_epoch=%d.txt" % epoch
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_train_cvsp + filename, acc_list, fmt="%d")

                dir_train_cvsp_ratio = dir_labels + \
                    "training/training_data%d/CorrectVsPredictRatio/" % count
                sutil.MakeDirs(dir_train_cvsp_ratio)

                filename = "training_CorrectVsPredictRatio_epoch=%d.txt" % epoch
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_train_cvsp_ratio + filename, acc_list_ratio)

            acc_sum_train = np.sum(acc_list_all, axis=0)

            acc_balance_train = balanced_accuracy_score(
                labels_target_all_train, prediction_target_all_train)

            accuracy_balance_train[epoch] = acc_balance_train

            dir_train_summary = dir_labels + "training/training_summary/CorrectVsPredict/"
            sutil.MakeDirs(dir_train_summary)
            filename = "training_CorrectVsPredictAll_epoch=%d.txt" % epoch
            if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                np.savetxt(dir_train_summary + filename,
                           acc_sum_train, fmt="%d")

            train_loss_average = th.mean(history_train[epoch, :])
            train_acc_average = np.mean(accuracy_train[epoch, :])

            if epoch % display_rate == 0:

                print("epoch:%d, train loss:%f, train acc:%f" %
                      (epoch, train_loss_average, acc_balance_train))

            ############# test loss & accuracy################

            acc_list_all = np.zeros((n_eval, n_label, n_label))

            labels_target_all_test = np.array([])
            prediction_target_all_test = np.array([])

            for count, g_path in enumerate(test_path_list):

                g_eval = test_data_load[count]

                labels_eval = g_eval.nodes[final_time].data['celltype']

                cellID_file_path = cellID_test_path_list[count]
                cellID = np.loadtxt(cellID_file_path).astype(np.int64)

                if gpu != -1:
                    cellID = th.from_numpy(cellID).to(device)

                logits_eval = model(g_eval)
                loss_eval = criterion(logits_eval[cellID], labels_eval[cellID])
                history_eval[epoch, count] = loss_eval.data

                prediction_eval = logits_eval.data.max(1)[1]  # 予想ラベル

                labels_target = labels_eval[cellID]
                prediction_target = prediction_eval[cellID]

                if gpu != -1:
                    labels_target = labels_target .to("cpu")
                    prediction_target = prediction_target .to("cpu")

                labels_target_all_test = np.append(
                    labels_target_all_test, labels_target)
                prediction_target_all_test = np.append(
                    prediction_target_all_test, prediction_target)

                accuracy = prediction_target.eq(
                    labels_target).sum().numpy() / len(labels_target)  # 正解率

                accuracy_eval[epoch, count] = accuracy

                dir_train = dir_labels + "test/test_data%d/" % count
                dir_train_predictioin = dir_labels + "test/test_data%d/prediction/" % count
                sutil.MakeDirs(dir_train_predictioin)
                filename = "test_PredLabel_target_epoch=%d.txt" % epoch
                prediction_target_np = prediction_target.to(
                    'cpu').detach().numpy().copy()
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_train_predictioin +
                               filename, prediction_target_np)

                dir_train_correct = dir_labels + "test/test_data%d/correct/" % count
                sutil.MakeDirs(dir_train_correct)
                filename = "test_CorrLabel_target_epoch=%d.txt" % epoch
                labels_target_np = labels_target.to(
                    'cpu').detach().numpy().copy()
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_train_correct + filename, labels_target_np)

                dir_test_logits_all = dir_labels + "test/test_data%d/Logits_all/" % count
                sutil.MakeDirs(dir_test_logits_all)

                filename = "test_logits_all_epoch=%d.txt" % epoch
                logits_eval_np = logits_eval.to('cpu').detach().numpy().copy()
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_test_logits_all + filename, logits_eval_np)

                filename = "labels_all.txt"
                labels_np = labels.to('cpu').detach().numpy().copy()
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_train + filename, labels_np, fmt="%d")

                filename = "cellID_target.txt"
                if gpu != -1:
                    cellID_np = cellID.to('cpu').detach().numpy().copy()
                    if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                        np.savetxt(dir_train + filename, cellID_np, fmt="%d")
                else:
                    if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                        np.savetxt(dir_train + filename, cellID, fmt="%d")

                # Make chart of  CorrectVsPredict

                acc_list = np.zeros((n_label, n_label))
                acc_list_ratio = np.zeros((n_label, n_label))
                for i in range(n_label):

                    pred_temp = prediction_target_np[np.where(
                        labels_target_np == i)]

                    len1 = len(pred_temp)

                    for j in range(n_label):

                        len2 = len(pred_temp[np.where(pred_temp == j)])

                        acc_list[i, j] = len2
                        if len1 != 0:
                            acc_list_ratio[i, j] = len2/len1

                acc_list_all[count, :, :] = acc_list

                dir_test_cvsp = dir_labels + "test/test_data%d/CorrectVsPredict/" % count
                sutil.MakeDirs(dir_test_cvsp)
                filename = "test_CorrectVsPredict_epoch=%d.txt" % epoch
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_test_cvsp + filename, acc_list, fmt="%d")

                dir_test_cvsp_ratio = dir_labels + \
                    "test/test_data%d/CorrectVsPredictRatio/" % count
                sutil.MakeDirs(dir_test_cvsp_ratio)

                filename = "test_CorrectVsPredictRatio_epoch=%d.txt" % epoch
                if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                    np.savetxt(dir_test_cvsp_ratio + filename, acc_list_ratio)

            acc_sum_test = np.sum(acc_list_all, axis=0)

            acc_balance_test = balanced_accuracy_score(
                labels_target_all_test, prediction_target_all_test)

            accuracy_balance_eval[epoch] = acc_balance_test

            dir_test_summary = dir_labels + "test/test_summary/CorrectVsPredict/"
            sutil.MakeDirs(dir_test_summary)
            filename = "test_CorrectVsPredictAll_epoch=%d.txt" % epoch
            if (epoch % model_save_rate == 0) or (epoch == (epoch_total-1)):
                np.savetxt(dir_test_summary + filename, acc_sum_test, fmt="%d")

            test_loss_average = th.mean(history_eval[epoch, :])
            test_acc_average = np.mean(accuracy_eval[epoch, :])

            if (epoch % display_rate == 0) or (epoch == (epoch_total-1)):
                print("epoch:%d, test loss:%f, test acc:%f" %
                      (epoch, test_loss_average, acc_balance_test))

        ############# training################
        model.train()

        for count, g_path in enumerate(training_path_list):

            g = train_data_load[count]

            labels = g.nodes[final_time].data['celltype']

            cellID_file_path = cellID_training_path_list[count]
            cellID = np.loadtxt(cellID_file_path).astype(np.int64)

            if num_even == 1:
                cellID = np.array(cellID_list_all[count]).astype(np.int64)

            if gpu != -1:
                cellID = th.from_numpy(cellID).to(device)

            for i in range(iter_n):

                logits = model(g)

                loss = criterion(logits[cellID], labels[cellID])

                opt.zero_grad()

                loss.backward(retain_graph=False)

                if cuda_clear == 1:
                    del loss
                    th.cuda.empty_cache()

                opt.step()

            if null_net == 1:

                g = train_data_null[count]

                cellID_file_path = cellID_training_path_list[count]
                cellID = np.loadtxt(cellID_file_path).astype(np.int64)

                if num_even == 1:
                    cellID = np.array(cellID_list_all[count]).astype(np.int64)

                size_cellID = len(cellID)

                if gpu != -1:
                    cellID = th.from_numpy(cellID).to(device)

                score_tensor = th.ones(size_cellID, 3)

                if null_weight == 1:
                    score_tensor[:, 0] = score_tensor[:, 0] * weight_c0
                    score_tensor[:, 1] = score_tensor[:, 1] * weight_c1
                    score_tensor[:, 2] = score_tensor[:, 2] * weight_c2

                if null_weight == 0:
                    score_tensor[:, 0] = score_tensor[:, 0] * (1/3)
                    score_tensor[:, 1] = score_tensor[:, 1] * (1/3)
                    score_tensor[:, 2] = score_tensor[:, 2] * (1/3)

                if gpu != -1:
                    score_tensor = score_tensor.to(device)

                for i in range(iter_n):

                    logits = model(g)
                    logits = F.softmax(logits, dim=1)

                    loss2 = criterion2(logits[cellID], score_tensor)

                    opt.zero_grad()

                    loss2.backward(retain_graph=False)

                    if cuda_clear == 1:
                        del loss2
                        th.cuda.empty_cache()

                    opt.step()

    # Save the history of loss, balanced acc, weight for cross-entropy-loss
    if gpu != -1:
        history_train = history_train.to("cpu").detach().numpy().copy()
        history_eval = history_eval.to("cpu").detach().numpy().copy()

    np.savetxt(dir_log+'/history_loss.txt', history_train)
    np.savetxt(dir_log+'/history_eval_loss.txt', history_eval)

    np.savetxt(dir_log+'/history_acc_balanced_train.txt',
               accuracy_balance_train)
    np.savetxt(dir_log+'/history_acc_balanced_eval.txt', accuracy_balance_eval)

    return history_train, history_eval, accuracy_balance_train, accuracy_balance_eval, acc_sum_train, acc_sum_test


# Show Confusion matrix of the obtained model
def PrintConfusionMatrix(acc_sum_train, acc_sum_test, dir_log):
    print("Confusion matrix (Correct label vs predicted label)")
    print("0: NB,1: Del, 2: Div\n")
    print("Training: acc_summary")
    print((acc_sum_train[0, 0]+acc_sum_train[1, 1] +
          acc_sum_train[2, 2])/np.sum(acc_sum_train))
    print(acc_sum_train)

    total_train = np.array([np.sum(acc_sum_train[0, :]), np.sum(
        acc_sum_train[1, :]), np.sum(acc_sum_train[2, :])])
    num_labels_total_train_filepath = dir_log + "/num_labels_total_train.txt"
    np.savetxt(num_labels_total_train_filepath, total_train,
               header="0: None,1: dif, 2:div, total")

    print("\n")
    print("Test: acc_summary")
    print((acc_sum_test[0, 0]+acc_sum_test[1, 1] +
          acc_sum_test[2, 2])/np.sum(acc_sum_test))
    print(acc_sum_test)

    total_test = np.array([np.sum(acc_sum_test[0, :]), np.sum(
        acc_sum_test[1, :]), np.sum(acc_sum_test[2, :])])
    num_labels_total_test_filepath = dir_log + "/num_labels_total_test.txt"
    np.savetxt(num_labels_total_test_filepath, total_test,
               header="0: None,1: dif, 2:div, total")


# +

if __name__ == '__main__':
    time_start = time.time()

    history_train, history_eval, accuracy_balance_train, accuracy_balance_eval, acc_sum_train, acc_sum_test = RunTrainingCellFateGNN(model, training_path_list, test_path_list,
                                                                                                                                     cellID_training_path_list, cellID_test_path_list,
                                                                                                                                     train_data_load, test_data_load, train_data_null,
                                                                                                                                     n_label, device, parameters_set, final_time,
                                                                                                                                     dir_model, dir_labels, dir_log)

    elapsed_time = time.time()-time_start
    time_rec = "elapsed_time:{0}".format(elapsed_time) + "[sec]"
    print(time_rec)
    time_filename = dir_log + "/time.txt"
    file_time = open(time_filename, "w")
    file_time.write(time_rec)
    file_time.close()

    PrintConfusionMatrix(acc_sum_train, acc_sum_test, dir_log)

    gnn_models.PlotLossFunction(history_train, history_eval, epoch_total,
                                dir_fig, dir_log, figx, figy, left, right, bottom, top, dpi)
    gnn_models.PlotBalancedAccuracy(accuracy_balance_train, accuracy_balance_eval,
                                    epoch_total, dir_fig, dir_log, figx, figy, left, right, bottom, top, dpi)
    gnn_models.CalculateROCCurves(test_path_list, epoch_total, fate_label,
                                  dir_labels, result_dir_path, figx, figy, left, right, bottom, top, dpi)

    cm_name = "magma"
    n_color = 3
    max_index = 200
    cmap = plot_module.make_cmlist_to_dense(cm_name, n_color, max_index)

    gnn_models.PlotAUCTimeEvolution(
        p_abs, cmap, figx, figy, left, right, bottom, top, dpi)


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

""" GNN models and functions for plotting the performance """

import glob
import random
from itertools import cycle
import sklearn
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax
import matplotlib.pyplot as plt
from functions import plot_module as plot_module
from functions import system_utility as sutil
import torch as th
import torch.nn as nn

import dgl.function as fn
import torch.nn.functional as F

import numpy as np

import sys
import os



# -


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

    reg = param["reg"]
    norm_final = param["norm_final"]

    return hid_node, time_list, p_hidden, in_plane, NoSelfInfo, n_layers, skip, input_size, epoch_total, network_name, model_save_rate, average_switch, architecture, feature_list_edge, feature_edge_concat, input_size_edge, edge_switch, crop_width_list_train, crop_height_list_train, crop_width_list_test, crop_height_list_test, cuda_clear, reg, norm_final, in_feats, feature_self, feature, reg, norm_final


# Bidirectional GNN model
# Ver5
class CellFateNetTimeReversal(nn.Module):

    def __init__(self, input_size, in_feats, hid_feats, num_celltype, feature, time_set, p_hidden, in_plane, NoSelfInfo, feature_self, n_layers, skip, edge_switch, input_size_edge, feature_edge_concat, average_switch):
        super().__init__()

        self.version = 5

        self.input_size = input_size  # input_size: size of input vector
        self.in_feats = in_feats  # in_feats: size of encoded node vector
        self.hid_feats = hid_feats  # hid_feats: # of nodes in the MLP layer
        # num_celltype: # of cell types = len([NB,Del,Div])
        self.num_celltype = num_celltype
        self.feature = feature  # concatenated feature: feature to be learned
        # list of number of time plane: [0,1,2,3] for 4 time model
        self.time_set = time_set
        self.p_hidden = p_hidden  # dropout rate
        self.in_plane = in_plane  # number of iteration of spatial message passing
        # if 1, cell external model. if 0, full model.
        self.NoSelfInfo = NoSelfInfo
        # concatenated feature for the taget cell.
        self.feature_self = feature_self
        self.n_layers = n_layers  # number of hidden layers
        # if 0, sum aggregation. if 1, mean aggretation
        self.average_switch = average_switch

        # Functions we didn't use in the paper.
        # if 1, with skip connection. if 0, no skip connection.
        self.skip = skip
        # if 1, the model takes in edge features.
        self.edge_switch = edge_switch
        # if the size of the edge features.
        self.input_size_edge = input_size_edge
        # name of concatenated edge features
        self.feature_edge_concat = feature_edge_concat

        # Encoding (Note that we didn't use encoding layer in the paper.)
        fc_encode = []
        for i in range(self.n_layers+1):
            if i == 0:
                fc_encode.append(nn.Linear(self.input_size, self.hid_feats))
                fc_encode.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_encode.append(nn.Linear(self.hid_feats, self.in_feats))
            else:
                fc_encode.append(nn.Linear(self.hid_feats, self.hid_feats))
                fc_encode.append(nn.Dropout(p=self.p_hidden))

        self.fc_encode = nn.ModuleList(fc_encode)

        # backward temporal edge model

        fc_time_rev = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_time_rev.append(
                        nn.Linear(self.in_feats*4, self.hid_feats))
                elif self.skip == 0:
                    fc_time_rev.append(
                        nn.Linear(self.in_feats*2, self.hid_feats))

                fc_time_rev.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_time_rev.append(nn.Linear(self.hid_feats, self.in_feats))
            else:
                fc_time_rev.append(nn.Linear(self.hid_feats, self.hid_feats))
                fc_time_rev.append(nn.Dropout(p=self.p_hidden))

        self.fc_time_rev = nn.ModuleList(fc_time_rev)

        # backward temporal node model

        fc_time_rev_nodemodel = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_time_rev_nodemodel.append(
                        nn.Linear(self.in_feats*3, self.hid_feats))
                if self.skip == 0:
                    fc_time_rev_nodemodel.append(
                        nn.Linear(self.in_feats*2, self.hid_feats))

                fc_time_rev_nodemodel.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_time_rev_nodemodel.append(
                    nn.Linear(self.hid_feats, self.in_feats))
            else:
                fc_time_rev_nodemodel.append(
                    nn.Linear(self.hid_feats, self.hid_feats))
                fc_time_rev_nodemodel.append(nn.Dropout(p=self.p_hidden))

        self.fc_time_rev_nodemodel = nn.ModuleList(fc_time_rev_nodemodel)

        # spatial edge model

        fc_interaction = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_interaction.append(
                        nn.Linear(self.in_feats*4, self.hid_feats))
                elif self.skip == 0:
                    if self.edge_switch == 0:
                        fc_interaction.append(
                            nn.Linear(self.in_feats*2, self.hid_feats))
                    elif self.edge_switch == 1:
                        fc_interaction.append(
                            nn.Linear(self.in_feats*2 + self.input_size_edge, self.hid_feats))

                fc_interaction.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_interaction.append(nn.Linear(self.hid_feats, self.in_feats))
            else:
                fc_interaction.append(
                    nn.Linear(self.hid_feats, self.hid_feats))
                fc_interaction.append(nn.Dropout(p=self.p_hidden))

        self.fc_interaction = nn.ModuleList(fc_interaction)

        # spatial node model
        fc_nodemodel = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_nodemodel.append(
                        nn.Linear(self.in_feats*3, self.hid_feats))
                elif self.skip == 0:
                    fc_nodemodel.append(
                        nn.Linear(self.in_feats*2, self.hid_feats))

                fc_nodemodel.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_nodemodel.append(nn.Linear(self.hid_feats, self.in_feats))
            else:
                fc_nodemodel.append(nn.Linear(self.hid_feats, self.hid_feats))
                fc_nodemodel.append(nn.Dropout(p=self.p_hidden))

        self.fc_nodemodel = nn.ModuleList(fc_nodemodel)

        # forward temporal edge model

        fc_time = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_time.append(nn.Linear(self.in_feats*4, self.hid_feats))
                elif self.skip == 0:
                    fc_time.append(nn.Linear(self.in_feats*2, self.hid_feats))

                fc_time.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_time.append(nn.Linear(self.hid_feats, self.in_feats))
            else:
                fc_time.append(nn.Linear(self.hid_feats, self.hid_feats))
                fc_time.append(nn.Dropout(p=self.p_hidden))

        self.fc_time = nn.ModuleList(fc_time)

        # forward temporal node model

        fc_time_nodemodel = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_time_nodemodel.append(
                        nn.Linear(self.in_feats*3, self.hid_feats))
                if self.skip == 0:
                    fc_time_nodemodel.append(
                        nn.Linear(self.in_feats*2, self.hid_feats))

                fc_time_nodemodel.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_time_nodemodel.append(
                    nn.Linear(self.hid_feats, self.in_feats))
            else:
                fc_time_nodemodel.append(
                    nn.Linear(self.hid_feats, self.hid_feats))
                fc_time_nodemodel.append(nn.Dropout(p=self.p_hidden))

        self.fc_time_nodemodel = nn.ModuleList(fc_time_nodemodel)

        # decoder

        fc_output = []
        for i in range(self.n_layers+1):
            if i == 0:
                fc_output.append(nn.Linear(self.in_feats, self.hid_feats))

                fc_output.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_output.append(nn.Linear(self.hid_feats, self.num_celltype))
            else:
                fc_output.append(nn.Linear(self.hid_feats, self.hid_feats))
                fc_output.append(nn.Dropout(p=self.p_hidden))

        self.fc_output = nn.ModuleList(fc_output)

    def encode(self, graph, node, feature):

        x = self.fc_encode[0](graph.nodes[node].data[feature])

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_encode[2*i+1](x)
            x = self.fc_encode[2*i+2](x)
        return x

    def rev_outplane_edge_model(self, x):

        x = self.fc_time_rev[0](x)

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_time_rev[2*i+1](x)
            x = self.fc_time_rev[2*i+2](x)
        return x

    def rev_outplane_node_model(self, x):

        x = self.fc_time_rev_nodemodel[0](x)

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_time_rev_nodemodel[2*i+1](x)
            x = self.fc_time_rev_nodemodel[2*i+2](x)
        return x

    def inplane_edge_model(self, x):

        x = self.fc_interaction[0](x)

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_interaction[2*i+1](x)
            x = self.fc_interaction[2*i+2](x)
        return x

    def inplane_node_model(self, x):

        x = self.fc_nodemodel[0](x)

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_nodemodel[2*i+1](x)
            x = self.fc_nodemodel[2*i+2](x)
        return x

    def outplane_edge_model(self, x):

        x = self.fc_time[0](x)

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_time[2*i+1](x)
            x = self.fc_time[2*i+2](x)
        return x

    def outplane_node_model(self, x):

        x = self.fc_time_nodemodel[0](x)

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_time_nodemodel[2*i+1](x)
            x = self.fc_time_nodemodel[2*i+2](x)
        return x

    def decode(self, x):

        x = self.fc_output[0](x)

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_output[2*i+1](x)
            x = self.fc_output[2*i+2](x)
        return x

    def forward(self, graph):

        time_set = self.time_set

        len_time = len(time_set)

        feature_self = self.feature_self

        NoSelfInfo = self.NoSelfInfo

        skip = self.skip

        edge_switch = self.edge_switch
        feature_edge_concat = self.feature_edge_concat

        average_switch = self.average_switch

        feature = self.feature

        in_plane = self.in_plane

        enc = 'enc'
        dec = 'dec'

        enc_self = 'enc_self'

        with graph.local_scope():  # dgl.graph.ndata and edata is locally updated using dgl.graph.local_scope()

            ############ encoding #########

            for node in time_set:

                graph.nodes[node].data[enc] = self.encode(graph, node, feature)

            if NoSelfInfo == 1:  # ##Encoding the AllOne feature
                for node in time_set:

                    graph.nodes[node].data[enc_self] = self.encode(
                        graph, node, feature_self)

            if NoSelfInfo == 0:
                out_node_output_src = feature
                out_node_output_dst = feature
                enc_src = enc
                enc_dst = enc

            if NoSelfInfo == 1:
                out_node_output_src = feature
                out_node_output_dst = feature_self
                enc_src = enc
                enc_dst = enc_self

            ###################  backward temporal model ################
            # We need to perform bacward temporal edge (len_time-1) times

            ######## IMPORTANT: value doesn't change for the final layer because backward temporal node model doesn't update the layer########
            ######## Set default values as the original value #######
            for i in range(len_time-1):
                out_node_new = 'ROPNM_out%d' % (i+1)
                for count_node in range(len_time):
                    graph.nodes[time_set[count_node]
                                ].data[out_node_new] = graph.nodes[time_set[count_node]].data[out_node_output_dst]

            for i in range(len_time-1):
                # print(i

                ######### backward temporal edge model ############
                for count_node in reversed(range(len_time-1)):

                    rel = (time_set[count_node+1],
                           'time_rev', time_set[count_node])

                    out_edge_concat = 'concat_ROPEM%d' % (i+1)
                    out_edge_new = 'ROPEM_out%d' % (i+1)

                    if skip == 1:
                        graph.apply_edges(lambda edges: {out_edge_concat: th.cat(
                            [edges.src[out_node_output_src], edges.dst[out_node_output_dst], edges.src[enc_src], edges.dst[enc_dst]], dim=1)}, etype=rel)
                    elif skip == 0:
                        graph.apply_edges(lambda edges: {out_edge_concat: th.cat(
                            [edges.src[out_node_output_src], edges.dst[out_node_output_dst]], dim=1)}, etype=rel)

                    tmp1 = graph.edges[rel].data[out_edge_concat]

                    graph.edges[rel].data[out_edge_new] = self.rev_outplane_edge_model(
                        tmp1)

                # temporal message passing

                for count_node in reversed(range(len_time-1)):

                    rel = (time_set[count_node+1],
                           'time_rev', time_set[count_node])

                    out_edge_sum = 'ROPEM_sum%d' % (i+1)
                    out_medium = 'Rtime_m%d' % (i+1)
                    graph.send_and_recv(graph[rel].edges(), fn.copy_e(
                        out_edge_new, out_medium), fn.sum(out_medium, out_edge_sum), etype=rel)

                ######### backward temporal node model ############
                for count_node in reversed(range(len_time-1)):
                    # no time_sum1 for the final frame
                    time_name = time_set[count_node]

                    out_node_concat = 'concat_ROPNM%d' % (i+1)
                    out_node_new = 'ROPNM_out%d' % (i+1)

                    #out_node_input = 'IPNM_out%d'%(i+1)

                    # print(out_edge_sum)
                    # print(graph.nodes[time_name].data[out_edge_sum])
                    if skip == 1:

                        graph.apply_nodes(lambda nodes: {out_node_concat: th.cat(
                            [nodes.data[out_node_output_dst], nodes.data[out_edge_sum], nodes.data[enc_dst]], dim=1)}, ntype=time_name)
                    elif skip == 0:
                        graph.apply_nodes(lambda nodes: {out_node_concat: th.cat(
                            [nodes.data[out_node_output_dst], nodes.data[out_edge_sum]], dim=1)}, ntype=time_name)

                    tmp5 = graph.nodes[time_name].data[out_node_concat]

                    graph.nodes[time_name].data[out_node_new] = self.rev_outplane_node_model(
                        tmp5)

                out_node_output_src = out_node_new
                out_node_output_dst = out_node_new

            # Defin new feature name for the next model
            in_node_output_src = out_node_new
            in_node_output_dst = out_node_new

            for j in range(in_plane):
                ######### spatial edge model ############
                for node in time_set:

                    rel = (node, 'interaction', node)

                    in_edge_concat = 'concat_IPEM%d' % (j+1)
                    in_edge_new = 'IPEM_out%d' % (j+1)
                    # Define concatenated edge feature
                    if skip == 1:
                        graph.apply_edges(lambda edges: {in_edge_concat: th.cat(
                            [edges.src[in_node_output_src],  edges.dst[in_node_output_dst], edges.src[enc_src], edges.dst[enc_dst]], dim=1)}, etype=rel)
                    elif skip == 0:
                        # graph.apply_edges(lambda edges: {in_edge_concat : th.cat([edges.src[in_node_output_src],  edges.dst[in_node_output_dst]], dim=1)},etype = rel
                        if edge_switch == 0:
                            graph.apply_edges(lambda edges: {in_edge_concat: th.cat(
                                [edges.src[in_node_output_src],  edges.dst[in_node_output_dst]], dim=1)}, etype=rel)
                        elif edge_switch == 1:
                            graph.apply_edges(lambda edges: {in_edge_concat: th.cat(
                                [edges.src[in_node_output_src],  edges.dst[in_node_output_dst], edges.data[feature_edge_concat]], dim=1)}, etype=rel)

                    concat_0 = graph.edges[rel].data[in_edge_concat]

                    # Define concatenated edge feature
                    graph.edges[rel].data[in_edge_new] = self.inplane_edge_model(
                        concat_0)

                ############# spatial Message passing & aggregation summation as edge_sum1  ###########

                for node in time_set:

                    rel = (node, 'interaction', node)
                    in_edge_sum = 'IPEM_sum%d' % (j+1)
                    in_medium = 'm%d' % (j+1)

                    if average_switch != 1:  # 20210326
                        graph.send_and_recv(graph[rel].edges(), fn.copy_e(
                            in_edge_new, in_medium), fn.sum(in_medium, in_edge_sum), etype=rel)
                    elif average_switch == 1:  # 20210326
                        graph.send_and_recv(graph[rel].edges(), fn.copy_e(
                            in_edge_new, in_medium), fn.mean(in_medium, in_edge_sum), etype=rel)

                ######### spatial node model ############
                # UPDATE
                for time_name in time_set:

                    in_node_concat = 'concat_IPNM%d' % (j+1)
                    in_node_new = 'IPNM_out%d' % (j+1)

                    if skip == 1:
                        graph.apply_nodes(lambda nodes: {in_node_concat: th.cat(
                            [nodes.data[in_node_output_dst], nodes.data[in_edge_sum], nodes.data[enc_dst]], dim=1)}, ntype=time_name)
                    elif skip == 0:
                        graph.apply_nodes(lambda nodes: {in_node_concat: th.cat(
                            [nodes.data[in_node_output_dst], nodes.data[in_edge_sum]], dim=1)}, ntype=time_name)

                    concat_node_0 = graph.nodes[time_name].data[in_node_concat]

                    graph.nodes[time_name].data[in_node_new] = self.inplane_node_model(
                        concat_node_0)

                in_node_output_src = in_node_new
                in_node_output_dst = in_node_new

            # Defin new feature name for the next model
            if in_plane != 0:  # if with spatial interaction

                out_node_output = in_node_output_dst

            if in_plane == 0:  # if without spatial interaction

                out_node_output = in_node_output_dst
                if NoSelfInfo == 1:
                    print("Error: NoSelfInfo must be 0 for in_plane = 0.")

            ###################  forward temporal model  ################

            for i in range(len_time-1):
                # print(i)

                ######### forward temporal edge model ############
                for count_node in range(len_time-1):

                    rel = (time_set[count_node], 'time',
                           time_set[count_node + 1])

                    out_edge_concat = 'concat_OPEM%d' % (i+1)
                    out_edge_new = 'OPEM_out%d' % (i+1)

                    if skip == 1:
                        graph.apply_edges(lambda edges: {out_edge_concat: th.cat(
                            [edges.src[out_node_output], edges.dst[out_node_output], edges.src[enc_src], edges.dst[enc_dst]], dim=1)}, etype=rel)
                    elif skip == 0:
                        graph.apply_edges(lambda edges: {out_edge_concat: th.cat(
                            [edges.src[out_node_output], edges.dst[out_node_output]], dim=1)}, etype=rel)

                    tmp1 = graph.edges[rel].data[out_edge_concat]

                    graph.edges[rel].data[out_edge_new] = self.outplane_edge_model(
                        tmp1)

                # message passing
                for count_node in range(len_time-1):

                    rel = (time_set[count_node], 'time',
                           time_set[count_node + 1])

                    out_edge_sum = 'OPEM_sum%d' % (i+1)
                    out_medium = 'time_m%d' % (i+1)
                    graph.send_and_recv(graph[rel].edges(), fn.copy_e(
                        out_edge_new, out_medium), fn.sum(out_medium, out_edge_sum), etype=rel)

                ######### forward temporal node model ############
                for count_node in range(len_time-1):
                    # no time_sum1 for the first frame
                    time_name = time_set[count_node+1]

                    out_node_concat = 'concat_OPNM%d' % (i+1)
                    out_node_new = 'OPNM_out%d' % (i+1)

                    if skip == 1:
                        graph.apply_nodes(lambda nodes: {out_node_concat: th.cat(
                            [nodes.data[out_node_output], nodes.data[out_edge_sum], nodes.data[enc_dst]], dim=1)}, ntype=time_name)
                    elif skip == 0:
                        graph.apply_nodes(lambda nodes: {out_node_concat: th.cat(
                            [nodes.data[out_node_output], nodes.data[out_edge_sum]], dim=1)}, ntype=time_name)

                    tmp5 = graph.nodes[time_name].data[out_node_concat]

                    graph.nodes[time_name].data[out_node_new] = self.outplane_node_model(
                        tmp5)

                ######## IMPORTANT: value doesn't change for the first layer because forward temporal node model doesn't update the layer########
                graph.nodes[time_set[0]].data[out_node_new] = graph.nodes[time_set[0]
                                                                          ].data[in_node_output_dst]

                out_node_output = out_node_new

            ######### Decoding ############
            time_name = time_set[-1]

            tmp = graph.nodes[time_name].data[out_node_output]

            graph.nodes[time_name].data[dec] = self.decode(tmp)

            return graph.nodes[time_name].data[dec]


# Unidirectional GNN model
# Ver23
class CellFateNet(nn.Module):

    def __init__(self, input_size, in_feats, hid_feats, num_celltype, feature, time_set, p_hidden, in_plane, NoSelfInfo, feature_self, n_layers, skip, average_switch, edge_switch, input_size_edge, feature_edge_concat, reg, norm_final):
        super().__init__()

        self.version = 23

        self.input_size = input_size  # input_size: size of input vector
        self.in_feats = in_feats  # in_feats: size of encoded node vector
        self.hid_feats = hid_feats  # hid_feats: # of nodes in the MLP layer
        self.num_celltype = num_celltype  # of cell types = len([NB,Del,Div])
        self.feature = feature  # concatenated feature: feature to be learned
        # list of number of time plane: [0,1,2,3] for 4 time model
        self.time_set = time_set
        self.p_hidden = p_hidden  # dropout rate
        self.in_plane = in_plane  # number of iteration of spatial message passing
        # if 1, cell external model. if 0, full model.
        self.NoSelfInfo = NoSelfInfo
        # concatenated feature for the target cell.
        self.feature_self = feature_self
        self.n_layers = n_layers  # number of hidden layers
        # if 0, sum aggregation. if 1, mean aggretation
        self.average_switch = average_switch

        # For a model with edge features. We don't use edge features in the paper.
        self.edge_switch = edge_switch
        self.input_size_edge = input_size_edge
        self.feature_edge_concat = feature_edge_concat

        # Option for regulaization. We don't use edge features in the paper.
        self.reg = reg
        self.norm_final = norm_final
        self.skip = skip

        # Encoding. We don't use edge features in the paper.
        fc_encode = []
        for i in range(self.n_layers+1):
            if i == 0:
                fc_encode.append(nn.Linear(self.input_size, self.hid_feats))
                fc_encode.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_encode.append(nn.Linear(self.hid_feats, self.in_feats))
            else:
                fc_encode.append(nn.Linear(self.hid_feats, self.hid_feats))
                fc_encode.append(nn.Dropout(p=self.p_hidden))

        self.fc_encode = nn.ModuleList(fc_encode)

        # Spatial edge model

        fc_interaction = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_interaction.append(
                        nn.Linear(self.in_feats*4, self.hid_feats))
                elif self.skip == 0:
                    if self.edge_switch == 0:
                        fc_interaction.append(
                            nn.Linear(self.in_feats*2, self.hid_feats))
                    elif self.edge_switch == 1:
                        fc_interaction.append(
                            nn.Linear(self.in_feats*2 + self.input_size_edge, self.hid_feats))
                if self.reg == 1:
                    fc_interaction.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_interaction.append(
                        nn.BatchNorm1d(self.hid_feats))  # test

                # elif self.reg == 1

                fc_interaction.append(nn.Dropout(p=self.p_hidden))

            elif i == self.n_layers:
                fc_interaction.append(nn.Linear(self.hid_feats, self.in_feats))
                if self.reg == 1:
                    fc_interaction.append(nn.LayerNorm(self.in_feats))
                elif self.reg == 2:
                    fc_interaction.append(nn.BatchNorm1d(self.in_feats))

            else:
                fc_interaction.append(
                    nn.Linear(self.hid_feats, self.hid_feats))
                if self.reg == 1:
                    fc_interaction.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_interaction.append(nn.BatchNorm1d(self.hid_feats))
                fc_interaction.append(nn.Dropout(p=self.p_hidden))

        self.fc_interaction = nn.ModuleList(fc_interaction)

        # Spatial node model
        fc_nodemodel = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_nodemodel.append(
                        nn.Linear(self.in_feats*3, self.hid_feats))
                elif self.skip == 0:
                    fc_nodemodel.append(
                        nn.Linear(self.in_feats*2, self.hid_feats))

                if self.reg == 1:
                    fc_nodemodel.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_nodemodel.append(nn.BatchNorm1d(self.hid_feats))

                fc_nodemodel.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_nodemodel.append(nn.Linear(self.hid_feats, self.in_feats))
                if self.reg == 1:
                    fc_nodemodel.append(nn.LayerNorm(self.in_feats))
                elif self.reg == 2:
                    fc_nodemodel.append(nn.BatchNorm1d(self.in_feats))

            else:
                fc_nodemodel.append(nn.Linear(self.hid_feats, self.hid_feats))
                if self.reg == 1:
                    fc_nodemodel.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_nodemodel.append(nn.BatchNorm1d(self.hid_feats))

                fc_nodemodel.append(nn.Dropout(p=self.p_hidden))

        self.fc_nodemodel = nn.ModuleList(fc_nodemodel)

        # forward temporal edge model
        fc_time = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_time.append(nn.Linear(self.in_feats*4, self.hid_feats))
                elif self.skip == 0:
                    fc_time.append(nn.Linear(self.in_feats*2, self.hid_feats))

                if self.reg == 1:
                    fc_time.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_time.append(nn.BatchNorm1d(self.hid_feats))

                fc_time.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_time.append(nn.Linear(self.hid_feats, self.in_feats))
                if self.reg == 1:
                    fc_time.append(nn.LayerNorm(self.in_feats))
                elif self.reg == 2:
                    fc_time.append(nn.BatchNorm1d(self.in_feats))
            else:
                fc_time.append(nn.Linear(self.hid_feats, self.hid_feats))
                if self.reg == 1:
                    fc_time.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_time.append(nn.BatchNorm1d(self.hid_feats))

                fc_time.append(nn.Dropout(p=self.p_hidden))

        self.fc_time = nn.ModuleList(fc_time)

        # forward temporal node model

        fc_time_nodemodel = []
        for i in range(self.n_layers+1):
            if i == 0:
                if self.skip == 1:
                    fc_time_nodemodel.append(
                        nn.Linear(self.in_feats*3, self.hid_feats))
                elif self.skip == 0:
                    fc_time_nodemodel.append(
                        nn.Linear(self.in_feats*2, self.hid_feats))

                if self.reg == 1:
                    fc_time_nodemodel.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_time_nodemodel.append(nn.BatchNorm1d(self.hid_feats))

                fc_time_nodemodel.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_time_nodemodel.append(
                    nn.Linear(self.hid_feats, self.in_feats))

                if self.reg == 1:
                    fc_time_nodemodel.append(nn.LayerNorm(self.in_feats))
                elif self.reg == 2:
                    fc_time_nodemodel.append(nn.BatchNorm1d(self.in_feats))

            else:
                fc_time_nodemodel.append(
                    nn.Linear(self.hid_feats, self.hid_feats))
                if self.reg == 1:
                    fc_time_nodemodel.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_time_nodemodel.append(nn.BatchNorm1d(self.hid_feats))
                fc_time_nodemodel.append(nn.Dropout(p=self.p_hidden))

        self.fc_time_nodemodel = nn.ModuleList(fc_time_nodemodel)

        # decoder

        fc_output = []
        for i in range(self.n_layers+1):
            if i == 0:
                fc_output.append(nn.Linear(self.in_feats, self.hid_feats))
                if self.reg == 1:
                    fc_output.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_output.append(nn.BatchNorm1d(self.hid_feats))

                fc_output.append(nn.Dropout(p=self.p_hidden))
            elif i == self.n_layers:
                fc_output.append(nn.Linear(self.hid_feats, self.num_celltype))
                if self.reg == 1:
                    fc_output.append(nn.LayerNorm(self.num_celltype))
                elif self.reg == 2:
                    fc_output.append(nn.BatchNorm1d(self.num_celltype))

            else:
                fc_output.append(nn.Linear(self.hid_feats, self.hid_feats))
                if self.reg == 1:
                    fc_output.append(nn.LayerNorm(self.hid_feats))
                elif self.reg == 2:
                    fc_output.append(nn.BatchNorm1d(self.hid_feats))
                fc_output.append(nn.Dropout(p=self.p_hidden))

        self.fc_output = nn.ModuleList(fc_output)

    def encode(self, graph, node, feature):

        x = self.fc_encode[0](graph.nodes[node].data[feature])

        for i in range(self.n_layers):
            x = F.relu(x)
            x = self.fc_encode[2*i+1](x)
            x = self.fc_encode[2*i+2](x)
        return x

    def inplane_edge_model(self, x):

        if self.reg == 0:
            x = self.fc_interaction[0](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_interaction[2*i+1](x)
                x = self.fc_interaction[2*i+2](x)

        if self.reg == 1 or self.reg == 2:
            x = self.fc_interaction[0](x)
            x = self.fc_interaction[1](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_interaction[3*i+2](x)
                x = self.fc_interaction[3*i+3](x)
                if i != (self.n_layers-1):
                    x = self.fc_interaction[3*i+4](x)
                elif self.norm_final == 1:
                    x = self.fc_interaction[3*i+4](x)

        return x

    def inplane_node_model(self, x):

        if self.reg == 0:
            x = self.fc_nodemodel[0](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_nodemodel[2*i+1](x)
                x = self.fc_nodemodel[2*i+2](x)

        if self.reg == 1 or self.reg == 2:
            x = self.fc_nodemodel[0](x)
            x = self.fc_nodemodel[1](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_nodemodel[3*i+2](x)
                x = self.fc_nodemodel[3*i+3](x)
                if i != (self.n_layers-1):
                    x = self.fc_nodemodel[3*i+4](x)
                elif self.norm_final == 1:
                    x = self.fc_nodemodel[3*i+4](x)

        return x

    def outplane_edge_model(self, x):

        if self.reg == 0:
            x = self.fc_time[0](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_time[2*i+1](x)
                x = self.fc_time[2*i+2](x)

        if self.reg == 1 or self.reg == 2:
            x = self.fc_time[0](x)
            x = self.fc_time[1](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_time[3*i+2](x)
                x = self.fc_time[3*i+3](x)
                if i != (self.n_layers-1):
                    x = self.fc_time[3*i+4](x)
                elif self.norm_final == 1:
                    x = self.fc_time[3*i+4](x)

        return x

    def outplane_node_model(self, x):

        if self.reg == 0:
            x = self.fc_time_nodemodel[0](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_time_nodemodel[2*i+1](x)
                x = self.fc_time_nodemodel[2*i+2](x)

        if self.reg == 1 or self.reg == 2:
            x = self.fc_time_nodemodel[0](x)
            x = self.fc_time_nodemodel[1](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_time_nodemodel[3*i+2](x)
                x = self.fc_time_nodemodel[3*i+3](x)
                if i != (self.n_layers-1):
                    x = self.fc_time_nodemodel[3*i+4](x)
                elif self.norm_final == 1:
                    x = self.fc_time_nodemodel[3*i+4](x)

        return x

    def decode(self, x):

        if self.reg == 0:
            x = self.fc_output[0](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_output[2*i+1](x)
                x = self.fc_output[2*i+2](x)

        if self.reg == 1 or self.reg == 2:
            x = self.fc_output[0](x)
            x = self.fc_output[1](x)
            for i in range(self.n_layers):
                x = F.relu(x)
                x = self.fc_output[3*i+2](x)
                x = self.fc_output[3*i+3](x)
                if i != (self.n_layers-1):
                    x = self.fc_output[3*i+4](x)
                elif self.norm_final == 1:
                    x = self.fc_output[3*i+4](x)

        return x

    def forward(self, graph):

        time_set = self.time_set

        len_time = len(time_set)

        feature_self = self.feature_self

        NoSelfInfo = self.NoSelfInfo

        skip = self.skip

        average_switch = self.average_switch

        edge_switch = self.edge_switch
        feature_edge_concat = self.feature_edge_concat

        feature = self.feature
        in_plane = self.in_plane

        enc = 'enc'
        dec = 'dec'

        enc_self = 'enc_self'

        with graph.local_scope():  # dgl.graph.ndata and edata is locally updated using dgl.graph.local_scope()

            ############ encoding #########

            for node in time_set:

                graph.nodes[node].data[enc] = self.encode(graph, node, feature)

            if NoSelfInfo == 1:  # ##Encoding the AllOne feature
                for node in time_set:

                    graph.nodes[node].data[enc_self] = self.encode(
                        graph, node, feature_self)

            if NoSelfInfo == 0:
                in_node_output_src = feature
                in_node_output_dst = feature
                enc_src = enc
                enc_dst = enc

            if NoSelfInfo == 1:
                in_node_output_src = feature
                in_node_output_dst = feature_self
                enc_src = enc
                enc_dst = enc_self

            for j in range(in_plane):
                ######### spatial edge model############
                for node in time_set:

                    rel = (node, 'interaction', node)

                    in_edge_concat = 'concat_IPEM%d' % (j+1)
                    in_edge_new = 'IPEM_out%d' % (j+1)
                    # Define concatenated edge feature
                    if skip == 1:
                        graph.apply_edges(lambda edges: {in_edge_concat: th.cat(
                            [edges.src[in_node_output_src],  edges.dst[in_node_output_dst], edges.src[enc_src], edges.dst[enc_dst]], dim=1)}, etype=rel)
                    elif skip == 0:
                        if edge_switch == 0:
                            graph.apply_edges(lambda edges: {in_edge_concat: th.cat(
                                [edges.src[in_node_output_src],  edges.dst[in_node_output_dst]], dim=1)}, etype=rel)
                        elif edge_switch == 1:
                            graph.apply_edges(lambda edges: {in_edge_concat: th.cat(
                                [edges.src[in_node_output_src],  edges.dst[in_node_output_dst], edges.data[feature_edge_concat]], dim=1)}, etype=rel)

                    concat_0 = graph.edges[rel].data[in_edge_concat]

                    # Define concatenated edge feature
                    graph.edges[rel].data[in_edge_new] = self.inplane_edge_model(
                        concat_0)

                ############# spatial Message passing & aggregation summation as edge_sum1  ###########

                for node in time_set:

                    rel = (node, 'interaction', node)
                    in_edge_sum = 'IPEM_sum%d' % (j+1)
                    in_medium = 'm%d' % (j+1)

                    if average_switch != 1:
                        graph.send_and_recv(graph[rel].edges(), fn.copy_e(
                            in_edge_new, in_medium), fn.sum(in_medium, in_edge_sum), etype=rel)
                    elif average_switch == 1:
                        graph.send_and_recv(graph[rel].edges(), fn.copy_e(
                            in_edge_new, in_medium), fn.mean(in_medium, in_edge_sum), etype=rel)

                ######### spatial node model############
                # UPDATE
                for time_name in time_set:

                    in_node_concat = 'concat_IPNM%d' % (j+1)
                    in_node_new = 'IPNM_out%d' % (j+1)

                    if skip == 1:
                        graph.apply_nodes(lambda nodes: {in_node_concat: th.cat(
                            [nodes.data[in_node_output_dst], nodes.data[in_edge_sum], nodes.data[enc_dst]], dim=1)}, ntype=time_name)
                    elif skip == 0:
                        graph.apply_nodes(lambda nodes: {in_node_concat: th.cat(
                            [nodes.data[in_node_output_dst], nodes.data[in_edge_sum]], dim=1)}, ntype=time_name)

                    concat_node_0 = graph.nodes[time_name].data[in_node_concat]

                    graph.nodes[time_name].data[in_node_new] = self.inplane_node_model(
                        concat_node_0)

                in_node_output_src = in_node_new  # this is stored to store t0 nodes
                in_node_output_dst = in_node_new  # this is stored to store t0 nodes

            if in_plane != 0:  # if with in-plane interaction

                out_node_output = in_node_output_dst

            if in_plane == 0:  # if without in-plane interaction

                out_node_output = in_node_output_dst
                if NoSelfInfo == 1:
                    print("Error: NoSelfInfo must be 0 for in_plane = 0.")

            ###################   temporal model  ################
            # We need to do forward temporal edge model (len_time-1) times

            for i in range(len_time-1):
                # print(i)

                ######### Forward temporal edge model ############
                for count_node in range(len_time-1):

                    rel = (time_set[count_node], 'time',
                           time_set[count_node + 1])

                    out_edge_concat = 'concat_OPEM%d' % (i+1)
                    out_edge_new = 'OPEM_out%d' % (i+1)

                    if NoSelfInfo == 0:  # added 20210116

                        if skip == 1:
                            graph.apply_edges(lambda edges: {out_edge_concat: th.cat(
                                [edges.src[out_node_output], edges.dst[out_node_output], edges.src[enc_src], edges.dst[enc_dst]], dim=1)}, etype=rel)
                        elif skip == 0:
                            graph.apply_edges(lambda edges: {out_edge_concat: th.cat(
                                [edges.src[out_node_output], edges.dst[out_node_output]], dim=1)}, etype=rel)
                    elif NoSelfInfo == 1:  # added 20210116

                        if skip == 1:
                            graph.apply_edges(lambda edges: {out_edge_concat: th.cat(
                                [edges.src[out_node_output], edges.dst[out_node_output], edges.src[enc_dst], edges.dst[enc_dst]], dim=1)}, etype=rel)
                        elif skip == 0:
                            graph.apply_edges(lambda edges: {out_edge_concat: th.cat(
                                [edges.src[out_node_output], edges.dst[out_node_output]], dim=1)}, etype=rel)

                    tmp1 = graph.edges[rel].data[out_edge_concat]

                    graph.edges[rel].data[out_edge_new] = self.outplane_edge_model(
                        tmp1)

                # temporal message passing
                for count_node in range(len_time-1):

                    rel = (time_set[count_node], 'time',
                           time_set[count_node + 1])

                    out_edge_sum = 'OPEM_sum%d' % (i+1)
                    out_medium = 'time_m%d' % (i+1)

                    graph.send_and_recv(graph[rel].edges(), fn.copy_e(
                        out_edge_new, out_medium), fn.sum(out_medium, out_edge_sum), etype=rel)

                ######### forward temporal node model############
                for count_node in range(len_time-1):
                    # no time_sum1 for the first frame
                    time_name = time_set[count_node+1]

                    out_node_concat = 'concat_OPNM%d' % (i+1)
                    out_node_new = 'OPNM_out%d' % (i+1)

                    if skip == 1:
                        graph.apply_nodes(lambda nodes: {out_node_concat: th.cat(
                            [nodes.data[out_node_output], nodes.data[out_edge_sum], nodes.data[enc_dst]], dim=1)}, ntype=time_name)
                    elif skip == 0:
                        graph.apply_nodes(lambda nodes: {out_node_concat: th.cat(
                            [nodes.data[out_node_output], nodes.data[out_edge_sum]], dim=1)}, ntype=time_name)

                    tmp5 = graph.nodes[time_name].data[out_node_concat]

                    graph.nodes[time_name].data[out_node_new] = self.outplane_node_model(
                        tmp5)

                ######## IMPORTANT: value doesn't change for the first layer because temporal node model doesn't update the layer########
                graph.nodes[time_set[0]].data[out_node_new] = graph.nodes[time_set[0]
                                                                          ].data[in_node_output_dst]

                out_node_output = out_node_new

            ######### Decoding ############
            time_name = time_set[-1]

            tmp = graph.nodes[time_name].data[out_node_output]

            graph.nodes[time_name].data[dec] = self.decode(tmp)

            return graph.nodes[time_name].data[dec]


def ConcatenateFeatures(training_path_list, test_path_list, time_list, feature, feature_self, dir_network, feature_list, device, gpu, null_net, NoSelfInfo, edge_switch, feature_list_edge):

    train_data_load = []
    test_data_load = []
    train_data_null = []

    for count, g_path in enumerate(training_path_list):

        g = sutil.PickleLoad(g_path)

        g_null = sutil.PickleLoad(g_path)

        if gpu != -1:
            g = g.to(device)
            g_null = g_null.to(device)

        ### Concatenate for node feature #####
        for node in time_list:

            tensor_list = []

            for feature_target in feature_list:

                if feature_target == 'zero_celltype_future_onehot2':
                    feature_target = 'celltype_future_onehot2'

                    tmp = g.nodes[node].data[feature_target]
                    tmp = th.zeros(tmp.shape).to(device)

                elif feature_target == 'random_feature':
                    tmp = g.nodes[node].data["zero"]
                    tmp = th.rand(tmp.shape).to(
                        device)  # generate random number

                else:
                    tmp = g.nodes[node].data[feature_target]

                tensor_list.append(tmp)

            g.nodes[node].data[feature] = th.cat(tensor_list, dim=1)
            tensor_size = g.nodes[node].data[feature].size()

            ## Define Null network ###
            if null_net == 1:
                g_null.nodes[node].data[feature] = th.tensor(
                    np.zeros((tensor_size[0], tensor_size[1])), dtype=th.float).to(device)
                if NoSelfInfo == 1:

                    tensor_size = g_null.nodes[node].data[feature].size()

                    if gpu != -1:

                        g_null.nodes[node].data[feature_self] = th.tensor(
                            np.zeros((tensor_size[0], tensor_size[1])), dtype=th.float).to(device)
                    else:

                        g_null.nodes[node].data[feature_self] = th.tensor(
                            np.zeros((tensor_size[0], tensor_size[1])), dtype=th.float)

            if NoSelfInfo == 1:

                tensor_size = g.nodes[node].data[feature].size()

                if gpu != -1:

                    g.nodes[node].data[feature_self] = th.tensor(
                        np.zeros((tensor_size[0], tensor_size[1])), dtype=th.float).to(device)
                else:

                    g.nodes[node].data[feature_self] = th.tensor(
                        np.zeros((tensor_size[0], tensor_size[1])), dtype=th.float)

        ### Concatenate for edge feature #####

        if edge_switch == 1:
            for node in time_list:

                edge_tuple = (node, "interaction", node)

                tensor_list_edge = []

                for feature_target_edge in feature_list_edge:

                    if feature_target_edge == 'zero_EdgeLength1_norm':
                        feature_target_edge = 'EdgeLength1_norm'

                        tmp = g.edges[edge_tuple].data[feature_target_edge]

                        tmp = th.ones(tmp.shape).to(device)

                    else:
                        tmp = g.edges[edge_tuple].data[feature_target_edge]

                    tensor_list_edge.append(tmp)

                # print(tensor_list_edge)
                g.edges[edge_tuple].data[feature_edge_concat] = th.cat(
                    tensor_list_edge, dim=1)

        sutil.PickleDump(g, dir_network + "train_%d.pickle" % count)

        train_data_load.append(g)

        if null_net == 1:
            sutil.PickleDump(g_null, dir_network +
                             "train_null_%d.pickle" % count)

            train_data_null.append(g_null)

    for count, g_path in enumerate(test_path_list):
        g_eval = sutil.PickleLoad(g_path)

        if gpu != -1:
            g_eval = g_eval.to(device)

        for node in time_list:

            tensor_list = []

            for feature_target in feature_list:

                if feature_target == 'zero_celltype_future_onehot2':
                    feature_target = 'celltype_future_onehot2'

                    tmp = g_eval.nodes[node].data[feature_target]
                    tmp = th.zeros(tmp.shape).to(device)

                elif feature_target == 'random_feature':
                    tmp = g_eval.nodes[node].data["zero"]
                    tmp = th.rand(tmp.shape).to(device)

                else:
                    tmp = g_eval.nodes[node].data[feature_target]

                tensor_list.append(tmp)

            g_eval.nodes[node].data[feature] = th.cat(tensor_list, dim=1)

            if NoSelfInfo == 1:
                tensor_size = g_eval.nodes[node].data[feature].size()

                if gpu != -1:

                    g_eval.nodes[node].data[feature_self] = th.tensor(
                        np.zeros((tensor_size[0], tensor_size[1])), dtype=th.float).to(device)
                else:

                    g_eval.nodes[node].data[feature_self] = th.tensor(
                        np.zeros((tensor_size[0], tensor_size[1])), dtype=th.float)

    ### Concatenate for edge feature #####

        if edge_switch == 1:
            for node in time_list:

                edge_tuple = (node, "interaction", node)

                tensor_list_edge = []

                for feature_target_edge in feature_list_edge:

                    if feature_target_edge == 'zero_EdgeLength1_norm':
                        feature_target_edge = 'EdgeLength1_norm'

                        tmp = g_eval.edges[edge_tuple].data[feature_target_edge]

                        tmp = th.ones(tmp.shape).to(device)

                    else:
                        tmp = g_eval.edges[edge_tuple].data[feature_target_edge]

                    tensor_list_edge.append(tmp)

                g_eval.edges[edge_tuple].data[feature_edge_concat] = th.cat(
                    tensor_list_edge, dim=1)

        sutil.PickleDump(g_eval, dir_network + "test_%d.pickle" % count)

        test_data_load.append(g_eval)

    return train_data_load, test_data_load, train_data_null


# +
# Calculate weight for cross-sensitive learning
# Weight is calculated from the training data set

def CalculateWeightTrainingData(training_path_list, cellID_training_path_list, final_time, penalty_zero):

    num0 = 0
    num1 = 0
    num2 = 0

    for count2, g_path in enumerate(training_path_list):

        g = sutil.PickleLoad(g_path)

        labels = g.nodes[final_time].data['celltype']

        cellID_file_path = cellID_training_path_list[count2]
        cellID = np.loadtxt(cellID_file_path).astype(np.int64)

        labels_target = labels[cellID]

        labels_np = labels_target.to('cpu').detach().numpy().copy()

        num0 = num0 + len(np.where(labels_np == 0)[0])
        num1 = num1 + len(np.where(labels_np == 1)[0])
        num2 = num2 + len(np.where(labels_np == 2)[0])

    weight_c0 = num0/(num0 + num1 + num2)
    weight_c1 = num1/(num0 + num1 + num2)
    weight_c2 = num2/(num0 + num1 + num2)

    # Weight for cross-entropy-loss
    weight_cs = th.tensor([1.0/(weight_c0*penalty_zero),
                          1.0/weight_c1, 1.0/weight_c2], dtype=th.float)
    print("weight_label")
    print(weight_cs)

    return num0, num1, num2, weight_cs, weight_c0, weight_c1, weight_c2


# -

# plot loss function
def PlotLossFunction(history_train, history_eval, epoch_total, dir_fig, dir_log, figx, figy, left, right, bottom, top, dpi):
    # print(feature_list)
    # print(p_abs)
    xscale_list = ["linear"]
    yscale_list = ["linear"]

    x = range(1, epoch_total+1)  # for log scale +1
    y = np.mean(history_train, axis=1)

    y_eval = np.mean(history_eval, axis=1)

    figname_base = "history_loss"
    xlabel = 'epoch'
    ylabel = 'loss'
    for xscale in xscale_list:
        for yscale in yscale_list:
            figname = dir_fig + "/" + figname_base + \
                "_xscale=%s_yscale=%s.pdf" % (xscale, yscale)
            figname_png = dir_fig + "/" + figname_base + \
                "_xscale=%s_yscale=%s.png" % (xscale, yscale)
            fig = plt.figure(figsize=(figx, figy))
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # plt.hold(True)
            ax.plot(x, y, c="r", label="train")
            ax.plot(x, y_eval, c="b", label="test")
            ax.legend()
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            title_name = figname_base + "_finalloss=%f" % (y[-1])
            ax.set_title(title_name)
            plt.savefig(figname)
            plt.savefig(figname_png, dpi=dpi)
            plt.show()

    plt.close()

    print("final training loss = %f" % y[-1])
    print("final test loss = %f" % y_eval[-1])

    fn = dir_log + "/final_loss_epoch.txt"
    np.savetxt(fn, [y[-1], x[-1]])

    fn = dir_log + "/final_eval_loss_epoch.txt"
    np.savetxt(fn, [y_eval[-1], x[-1]])


# plot balanced accuracy
def PlotBalancedAccuracy(accuracy_balance_train, accuracy_balance_eval, epoch_total, dir_fig, dir_log, figx, figy, left, right, bottom, top, dpi):
    # print(feature_list)
    # print(p_abs)
    xscale_list = ["linear"]
    yscale_list = ["linear"]

    x = range(1, epoch_total+1)
    y = accuracy_balance_train

    y_eval = accuracy_balance_eval

    # print(len(y))
    # print(y)

    figname_base = "history_acc_balanced"
    xlabel = 'epoch'
    ylabel = 'acc'
    for xscale in xscale_list:
        for yscale in yscale_list:
            figname = dir_fig + "/" + figname_base + \
                "_xscale=%s_yscale=%s.pdf" % (xscale, yscale)
            figname_png = dir_fig + "/" + figname_base + \
                "_xscale=%s_yscale=%s.png" % (xscale, yscale)
            fig = plt.figure(figsize=(figx, figy))
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
            ax.set_xlabel(xlabel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # plt.hold(True)
            ax.plot(x, y, c="r", label="train")
            ax.plot(x, y_eval, c="b", label="test")
            ax.legend()
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            title_name = figname_base + "_finalacc=%f" % (y[-1])
            ax.set_title(title_name)
            plt.savefig(figname)
            plt.savefig(figname_png, dpi=dpi)
            plt.show()

    plt.close()

    print("final training balanced accurcay = %f" % y[-1])
    print("final test balanced accuracy = %f" % y_eval[-1])

    fn = dir_log + "/final_acc_balanced_epoch.txt"
    np.savetxt(fn, [y[-1], x[-1]])

    fn = dir_log + "/final_eval_balanced_acc_epoch.txt"
    np.savetxt(fn, [y_eval[-1], x[-1]])


# +
################## calculate ROC curve for test data############
# Evaluation at the final epoch.
# Preprocessing
# Modified from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

def CalculateROCCurves(test_path_list, epoch_total, fate_label, dir_labels, result_dir_path, figx, figy, left, right, bottom, top, dpi):

    dir_ROC = result_dir_path + "ROC/"
    sutil.MakeDirs(dir_ROC)

    dir_ROC_figs = dir_ROC + "figs/"
    sutil.MakeDirs(dir_ROC_figs)

    dir_ROC_data = dir_ROC + "data/"
    sutil.MakeDirs(dir_ROC_data)

    lw = 2

    epoch = epoch_total-1

    class_list = list(range(len(fate_label)))

    ############# Test ################

    correct_label_all = np.empty((0), int)
    max_prob_all = np.empty(0)
    max_label_all = np.empty(0, int)

    prob_all = np.empty((0, 3))

    for count, g_path in enumerate(test_path_list):

        dir_test = dir_labels + "test/test_data%d/" % count

        dir_test_logits_all = dir_labels + "test/test_data%d/Logits_all/" % count

        filename = "test_logits_all_epoch=%d.txt" % epoch
        logits_eval_np = np.loadtxt(dir_test_logits_all + filename)
        filename = "cellID_target.txt"

        cellID_np = np.loadtxt(dir_test + filename, dtype=int)

        # correct labels for the target cells
        dir_train_correct = dir_labels + "test/test_data%d/correct/" % count
        filename = "test_CorrLabel_target_epoch=%d.txt" % epoch
        labels_target_np = np.loadtxt(
            dir_train_correct + filename, dtype="int8")

        logits_eval_target = logits_eval_np[cellID_np]

        prob = softmax(logits_eval_target, axis=1)

        max_prob = np.max(prob, axis=1)
        max_label = np.argmax(prob, axis=1)

        correct_label_all = np.append(
            correct_label_all, labels_target_np, axis=0)
        max_prob_all = np.append(max_prob_all, max_prob, axis=0)
        max_label_all = np.append(max_label_all, max_label, axis=0)

        prob_all = np.append(prob_all, prob, axis=0)

    ################## calculate ROC curve for test data############
    # Evaluation at the final epoch.
    # Draw ROC curve
    # To plot ROC curves for the multilabel problem, we modified an example code from
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    y_score = prob_all

    # Binarize the output   # y_test is correct label
    y_test = label_binarize(correct_label_all, classes=class_list)
    n_classes = y_test.shape[1]

    # print(y_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # print(y_test.ravel())
    # print(y_score.ravel())
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # print(all_fpr)

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

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of %s ' % fate_label[i] + '(area = %.02f)' % roc_auc[i])

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC to multi-class')
    plt.legend(loc="lower right", fontsize=10)

    figname = dir_ROC_figs + "ROC.pdf"
    plt.savefig(figname)
    figname_png = dir_ROC_figs + "ROC.png"
    plt.savefig(figname_png, dpi=dpi)

    # plt.figure(figsize=(6,6))
    plt.show()

    sutil.PickleDump(fpr, dir_ROC_data + "fpr.pickle")
    sutil.PickleDump(tpr, dir_ROC_data + "tpr.pickle")
    sutil.PickleDump(roc_auc, dir_ROC_data + "roc_auc.pickle")


# +

# Prepare a list of file names  For training

def ReturnListFilenamesNetworkAndID(base_path_list_training, crop_width_list_train, crop_height_list_train, network_num, num_time, omit_t0, shuffle, border):

    learning_path_list = []

    cellID_learning_path_list = []

    for i, base_path in enumerate(base_path_list_training):

        analdir_name = "analysis/"
        analdir_path = base_path + analdir_name

        crop_width = crop_width_list_train[i]
        crop_height = crop_height_list_train[i]

        network_base_dir = "network%s_crop_w=%d_h=%d/" % (
            network_num, crop_width, crop_height)

        if shuffle == 0:
            networkdir_name = network_base_dir + \
                "network%s_num_w=%d_h=%d_time=%d/" % (
                    network_num, crop_width, crop_height, num_time)
        if shuffle == 1:
            networkdir_name = network_base_dir + \
                "network%s_num_w=%d_h=%d_time=%d_shuffle/" % (
                    network_num, crop_width, crop_height, num_time)

        networkdir_path = analdir_path + networkdir_name

        files = os.listdir(networkdir_path)
        # print(files)

        files_in_train_tmp = [s for s in files if 'NetworkWithFeartures' in s]

        for j, file_tmp in enumerate(files_in_train_tmp):
            start = j
            end = j+num_time-1
            file = "NetworkWithFeartures_t=%dto%d.pickle" % (start, end)
            learning_path_list.append(networkdir_path + file)
            #print(networkdir_path+ file)

    # print(learning_path_list)

    for i, base_path in enumerate(base_path_list_training):

        analdir_name = "analysis/"
        analdir_path = base_path + analdir_name

        crop_width = crop_width_list_train[i]
        crop_height = crop_height_list_train[i]

        network_base_dir = "network%s_crop_w=%d_h=%d/" % (
            network_num, crop_width, crop_height)

        if border == 1:
            cellIDdir_name = network_base_dir + \
                "network%s_cellID_FinalLayer_num_w=%d_h=%d_time=%d/" % (
                    network_num, crop_width, crop_height, num_time)
        if border == 0:
            cellIDdir_name = network_base_dir + \
                "network%s_cellID_FinalLayer_noborder_num_w=%d_h=%d_time=%d/" % (
                    network_num, crop_width, crop_height, num_time)
        cellIDdir_path = analdir_path + cellIDdir_name

        files_tmp = os.listdir(cellIDdir_path)
        # print(files)

        for j, file_tmp in enumerate(files_tmp):
            start = j
            end = j+num_time-1
            file = "CellID_FinalLayer_t=%dto%d.txt" % (start, end)
            cellID_learning_path_list.append(cellIDdir_path + file)
            #print(cellIDdir_path+ file)

    # print(cellID_learning_path_list)

    if omit_t0 == 1:

        learning_path_list = learning_path_list[1:]
        cellID_learning_path_list = cellID_learning_path_list[1:]
    # print(learning_path_list)
    # print(cellID_learning_path_list)

    return learning_path_list, cellID_learning_path_list


# +

# Make CellID list with equal number of cells for each label. Weight is set to [3,3,3] as well. Only for training data


def MakeCellIDListWithEqualNumbers(num0, num1, num2, training_path_list, final_time, cellID_learning_path_list):
    num_label_min = np.min([num0, num1, num2])

    # print(num_label_min)

    weight_cs = th.tensor([3.0, 3.0, 3.0], dtype=th.float)

    list_sample0 = np.array(sorted(random.sample(
        list(range(num0)), num_label_min))).astype(np.int64)
    list_sample1 = np.array(sorted(random.sample(
        list(range(num1)), num_label_min))).astype(np.int64)
    list_sample2 = np.array(sorted(random.sample(
        list(range(num2)), num_label_min))).astype(np.int64)

    # print(list_sample0)

    num0_tmp = 0
    num1_tmp = 0
    num2_tmp = 0

    cellID_0_all = []
    cellID_1_all = []
    cellID_2_all = []

    for count2, g_path in enumerate(training_path_list):
        # print(g_path)

        # print(g_path)
        g = pickle_load(g_path)
        # print(g.nodes['t2'].data['celltype'])
        # print(g)
        labels = g.nodes[final_time].data['celltype']

        cellID_file_path = cellID_learning_path_list[count2]
        # print(cellID_file_path)
        cellID = np.loadtxt(cellID_file_path).astype(np.int64)

        labels_target = labels[cellID]

        labels_np = labels_target.to('cpu').detach().numpy().copy()

        # print(labels_np)

        num0_target = len(np.where(labels_np == 0)[0])
        num1_target = len(np.where(labels_np == 1)[0])
        num2_target = len(np.where(labels_np == 2)[0])

        cellID_tmp0 = cellID[np.where(labels_np == 0)]
        cellID_tmp1 = cellID[np.where(labels_np == 1)]
        cellID_tmp2 = cellID[np.where(labels_np == 2)]

        # print([num0_target,num1_target,num2_target])

        index_0 = list_sample0[np.where((list_sample0 >= num0_tmp) & (
            list_sample0 < num0_tmp+num0_target))[0]]-num0_tmp
        index_1 = list_sample1[np.where((list_sample1 >= num1_tmp) & (
            list_sample1 < num1_tmp+num1_target))[0]]-num1_tmp
        index_2 = list_sample2[np.where((list_sample2 >= num2_tmp) & (
            list_sample2 < num2_tmp+num2_target))[0]]-num2_tmp

        # print(index_0)
        # print(index_1)
        # print(index_2)

        cellID_0 = cellID_tmp0[index_0].tolist()

        cellID_1 = cellID_tmp1[index_1].tolist()

        cellID_2 = cellID_tmp2[index_2].tolist()

        cellID_0_all.append(cellID_0)
        cellID_1_all.append(cellID_1)
        cellID_2_all.append(cellID_2)

        # print(cellID_0.size)
        # print(cellID_1.size)

        # print(cellID_2.size)

        num0_tmp = num0_tmp+num0_target
        num1_tmp = num1_tmp+num1_target
        num2_tmp = num2_tmp+num2_target

    # Target cellIDs for each label
    # print(cellID_0_all)
    # print(cellID_1_all)
    # print(cellID_2_all)

    # Concatenate CellIDs for each network
    cellID_list_all = []
    for count2, g_path in enumerate(training_path_list):

        tmp_ID_list = cellID_0_all[count2]
        print(tmp_ID_list)
        tmp_ID_list.extend(cellID_1_all[count2])
        tmp_ID_list.extend(cellID_2_all[count2])

        cellID_list_all.append(tmp_ID_list)

    # print(cellID_list_all)

        # print(cellID_0_all[count2])
        # print(cellID_1_all[count2])
        # print(cellID_2_all[count2])

        # print(cellID_0_all[count2].size)
        # print(cellID_1_all[count2].size)
        # print(cellID_2_all[count2].size)

    return cellID_0_all, cellID_1_all, cellID_2_all, cellID_list_all


# +
###### Plot Maximum AUC #####


def PlotAUCTimeEvolution(p_abs, cmap, figx, figy, left, right, bottom, top, dpi):

    cm_name = "magma"
    n_color = 3
    max_index = 200
    cmap = plot_module.make_cmlist_to_dense(cm_name, n_color, max_index)

    base_dir = str(p_abs) + "/"

    # print(base_dir)
    parameter_path = base_dir + "parameters.pickle"

    param = sutil.PickleLoad(parameter_path)

    epoch = param["epoch_total"]

    model_rate_save = param["model_save_rate"]

    data_path = base_dir + "data_path/base_path_list_test.txt"

    data_path_test = sutil.Read1LineText(data_path)
    data_path_test = data_path_test.replace('\n', '')

    dir_ROC_time_data = base_dir + "ROC_time/data/"
    sutil.MakeDirs(dir_ROC_time_data)

    dir_ROC_time_figs = base_dir + "ROC_time/figs/"
    sutil.MakeDirs(dir_ROC_time_figs)

    files = glob.glob(base_dir + "labels/test/*")

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
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

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
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves

        fig = plt.figure(figsize=(figx*2, figy*2))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

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
        plt.close()

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

    plt.close()
    # plt.show()

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

""" Functions for calculating attribution"""

import torch.nn.functional as F
import numpy as np
import torch as th
import os
import sys
version = 1
print("version=%d" % version)


# -

def RemoveIsolatedNodesFromGraph(g, nodeID_list, time_list):
    # output
    # existing_ID: [[ID list of non-isolated cells at t0],[ID list of non-isolated cells at t1],,,]
    # g: graph which the isolated nodes are removed. Node IDs are re-indexed by preserving the orders.
    # new_index: index of nonisolated nodes in the cellIDs in the final layer.

    existing_ID = []
    for time_name in time_list:
        etype = (time_name, "interaction", time_name)
        #isolated_nodes = ((g.in_degrees(etype=etype) == 0) & (g.out_degrees(etype=etype) == 0)).nonzero().squeeze(1)

        isolated_nodes = ((g.in_degrees(etype=etype) == 0) & (
            g.out_degrees(etype=etype) == 0)).nonzero(as_tuple=False).squeeze(1)
        # print(time_name)
        #print((g.in_degrees(etype=etype) == 0))
        #print((g.out_degrees(etype=etype) == 0))
        #print(((g.in_degrees(etype=etype) == 0) & (g.out_degrees(etype=etype) == 0)).nonzero())
        # print(isolated_nodes)

        #nonisolated_nodes = ((g.in_degrees(etype=etype) != 0) | (g.out_degrees(etype=etype) != 0)).nonzero().squeeze(1)

        nonisolated_nodes = ((g.in_degrees(etype=etype) != 0) | (
            g.out_degrees(etype=etype) != 0)).nonzero(as_tuple=False).squeeze(1)
        g.remove_nodes(isolated_nodes, ntype=time_name)

        existing_ID.append(nonisolated_nodes)

        if time_name == time_list[-1]:  # For the final layer
            # print(nonisolated_nodes)
            nonisolated_nodes = nonisolated_nodes.to(
                'cpu').detach().numpy().copy()

            new_index = []
            for nodeID in nodeID_list:

                index = np.where(nonisolated_nodes == nodeID)[0]
                # print(np.where(nonisolated_nodes==nodeID))
                # print(index)
                new_index.append(index[0])

    return g, existing_ID, new_index


# +
# We used this in the paper

def compute_integrated_gradient_AllNode2(n, existing_IDList, input_original_exist, input_template_exist, input_blank_exist, input_original, model, time_list, feature, new_index, nodeID_list, ClassList):
    """
    Calculation of IG
    """
    # We don't rewrite input_original and input_blank, but use input_template to calculate the features parametrized by alpha(=i/n)

    total_node_number = len(new_index)

    total_class_number = len(ClassList)

    # score for all the alpha for all the nodes
    y_score_list_all = np.zeros((total_node_number, total_class_number, n+1))

    dIG_list_all = []  # See the figure in README.md
    #print("Start Calculate dIG")
    for i in range(0, n+1):  # Foward difference ## for calculation for (0,n) to obtain dF

        TargetFeature = tuple()
        for time_name in time_list:
            input_template_exist.nodes[time_name].data[feature] = input_original_exist.nodes[time_name].data[feature] * (
                i/n) + input_blank_exist.nodes[time_name].data[feature]
            input_template_exist.ndata[feature][time_name].requires_grad = True
            TargetFeature_tmp = (
                input_template_exist.nodes[time_name].data[feature],)

            TargetFeature += TargetFeature_tmp

        y = model(input_template_exist)
        y = F.softmax(y, dim=1)

        # print(y.shape)

        for ClassIndex in ClassList:
            # Keep the order of the original nodeID_list
            for ID_index, nodeID in enumerate(new_index):

                y_score = y[nodeID][ClassIndex]  # OK

                ag = th.autograd.grad(
                    y_score, TargetFeature, retain_graph=True)

                dIG_list = []  # See the figure in README.md
                for count, time_name in enumerate(time_list):
                    dy_list = ag[count]*(1/n)  # integrand at each step

                    # feature subtracted by the back gound
                    dx = input_original_exist.nodes[time_name].data[feature] - \
                        input_blank_exist.nodes[time_name].data[feature]

                    blank_tensor = input_original.nodes[time_name].data[feature]*0

                    index_exist = existing_IDList[count]

                    blank_tensor[index_exist] = dy_list * dx

                    dIG_list.append(blank_tensor)

                dIG_list_all.append(dIG_list)

                y_score_list_all[ID_index, ClassIndex, i] = y_score

                del y_score
                del ag
                th.cuda.empty_cache()

        del y
        th.cuda.empty_cache()

    #print("Start Calculate IG")
    # Calculate IG from dIG
    IG_list_all = []  # See the figure in README.md
    for ClassIndex in ClassList:
        for ID_index, nodeID in enumerate(nodeID_list):

            IG_list_forEachClassAndID = []
            for count, time_name in enumerate(time_list):

                # To calculate integrated gradient, prepare zero tensor
                IG_list_tmp = input_original.nodes[time_name].data[feature]*0
                for i in range(0, n):  # For integration we don't use i=n

                    index = i*total_class_number * \
                        (total_node_number) + ClassIndex * \
                        total_node_number + ID_index

                    dIG_list = dIG_list_all[index]

                    IG_list_tmp += dIG_list[count]

                IG_list_forEachClassAndID.append(IG_list_tmp)

            IG_list_all.append(IG_list_forEachClassAndID)

    return y_score_list_all, IG_list_all

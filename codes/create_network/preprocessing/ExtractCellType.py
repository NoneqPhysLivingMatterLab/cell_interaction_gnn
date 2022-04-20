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

"""Extract cell types"""

from functions import system_utility as sutil
import numpy as np
from scipy import io
import os
import sys
version = 2
print("version=%d" % version)



# %%
filename_yml = 'input_preprocessing.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

# Set matlab data mode or python data mode
# if npy actin file, npy = 1/ if mat actin file, npy =0
npy = yaml_obj["npy"]

crop_height = yaml_obj["crop_height"]
crop_width = yaml_obj["crop_width"]


# %%
####Load Tracking data####
base_path_file = "base_path.txt"
base_path = sutil.Read1LineText(base_path_file)
filename_txtfile = "segmentation_filename.txt"
filename = sutil.Read1LineText(filename_txtfile)

filename_txtfile2 = "lineage_filename.txt"
file_path2 = base_path + sutil.Read1LineText(filename_txtfile2)


if npy == 0:
    matdata2 = io.loadmat(file_path2)
    np_data = matdata2['TrackRec']
    # print(npy)

if npy == 1:

    np_data = np.load(file_path2)


# %%
#  Load mat file to numpy and neighbor pairs files
if npy == 0:
    file_path = base_path + filename
    matdata = io.loadmat(file_path)
    data = matdata['segmat'].astype(np.int64)


if npy == 1:
    file_path = base_path + filename
    matdata = np.load(file_path)
    data = matdata.astype(np.int64)


total_frame = data.shape[0]
height = data.shape[1]
width = data.shape[2]


analdir_name = "analysis/"
analdir_path = base_path + analdir_name
nlist_name = "neighbor_list/"
nlist_path = analdir_path + nlist_name


# %%
# export track data for each time

trackdir_name = "trackdata/"
trackdir_path = analdir_path + trackdir_name

sutil.MakeDirs(trackdir_path)

Timeframe = np_data[:, 0].astype(np.int64)

for frame_tmp in range(total_frame):

    frame = frame_tmp  # + 1 ### numbering start from 1 in mat file

    index = np.where(Timeframe == frame)

    data_target = np_data[index[0], :]

    filename = "trackdata_%d.txt" % frame_tmp
    np.savetxt(trackdir_path + filename, data_target,
               header="'Timeframe', 'CloneID', 'ParentID', 'CellID','PositionX','PositionY','VoronoiArea','G1MarkerInVoronoiArea','ActinSegmentationArea','G1MarkerInActinSegmentationArea' ")


# %%
# export cell state

# cell type, NB:0, Del: 1, Div: 2

celltypedir_name = "celltype/"
celltypedir_path = analdir_path + celltypedir_name

sutil.MakeDirs(celltypedir_path)

for frame_tmp in range(total_frame-1):

    filename = "trackdata_%d.txt" % frame_tmp
    current_data = np.loadtxt(trackdir_path + filename)

    cellID_c = current_data[:, 3].astype(np.int64)
    parent_c = current_data[:, 2].astype(np.int64)

    cell_num = current_data.shape[0]

    filename = "trackdata_%d.txt" % (frame_tmp+1)
    next_data = np.loadtxt(trackdir_path + filename)

    cellID_n = next_data[:, 3].astype(np.int64)
    parent_n = next_data[:, 2].astype(np.int64)

    # cell type list is in the order of current cellID   ### Set default cell type as del=1
    celltype = np.ones(cell_num)
    celltype = celltype.astype(np.int64)

    parent_n_list = parent_n.tolist()

    for i in parent_n_list:

        celltype[np.where(cellID_c == i)] = 2   # dividing cell

    for i in cellID_n.tolist():

        celltype[np.where(cellID_c == i)] = 0  # NB

    data_save = np.zeros((cell_num, 2))
    data_save[:, 0] = cellID_c
    data_save[:, 1] = celltype

    celltype_filename = "celltype_%d.txt" % frame_tmp

    np.savetxt(celltypedir_path + celltype_filename,
               data_save, header="CellID, Celltype")

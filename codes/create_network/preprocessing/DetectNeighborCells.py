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

"""Detect neighbors from segmentation npy file"""

import tifffile
from functions import system_utility as sutil
from scipy import io
import numpy as np
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
### For npy files ###
#### Load Segmentation data ####

if npy == 1:
    base_path_file = "base_path.txt"
    base_path = sutil.Read1LineText(base_path_file)
    filename_txtfile = "segmentation_filename.txt"
    filename = sutil.Read1LineText(filename_txtfile)
    file_path = base_path + filename
    matdata = np.load(file_path)
    data = matdata.astype(np.uint64)


# %%
### For mat files ###
#### Load Segmentation data ####
if npy == 0:
    base_path_file = "base_path.txt"
    base_path = sutil.Read1LineText(base_path_file)
    filename_txtfile = "segmentation_filename.txt"
    filename = sutil.Read1LineText(filename_txtfile)

    #  Load mat file to numpy
    file_path = base_path + filename
    matdata = io.loadmat(file_path)
    data = matdata['segmat'].astype(np.uint64)


# %%
#  Export label data as tiff

analdir_name = "analysis/"
analdir_path = base_path + analdir_name
sutil.MakeDirs(analdir_path)

tifdir_name = "tif/"
tifdir_path = analdir_path + tifdir_name
sutil.MakeDirs(tifdir_path)

total_frame = data.shape[0]
height = data.shape[1]
width = data.shape[2]
for frame in range(total_frame):
    img = data[frame]
    tifname = 'image_%d.tif' % frame
    tifffile.imsave(tifdir_path + tifname, img)

frame_total = total_frame
np.savetxt(base_path + "total_frame.txt", np.array([frame_total]), fmt="%d")


# %%
#  Export neighbor cells

bd_name = "boundary_neighbor/"
bddir_path = analdir_path + bd_name
sutil.MakeDirs(bddir_path)

nlist_name = "neighbor_list/"
nlist_path = analdir_path + nlist_name
sutil.MakeDirs(nlist_path)


for frame in range(total_frame):

    img = data[frame]

    uni_label = np.unique(img)

    boundary = np.where(img == 0)
    num_boundary = len(boundary[1])

    arr = np.zeros((0, 4))
    for i in range(num_boundary):

        row = boundary[0][i]
        col = boundary[1][i]

        if (row > 0) and (row < height-1) and (col > 0) and (col < width-1):

            img_crop = img[row-1: row+2, col-1: col+2]

            neighbor = np.unique(img_crop)

            if len(neighbor) == 3:  # For 2 cell boundaries
                n1 = neighbor[1]
                n2 = neighbor[2]

                add_list = np.array([row, col, n1, n2])

                arr = np.r_[arr, add_list.reshape(1, -1)]

            if len(neighbor) == 4:  # For tri-cellular junction

                n1 = neighbor[1]
                n2 = neighbor[2]
                n3 = neighbor[3]

                add_list = np.array([row, col, n1, n2])
                arr = np.r_[arr, add_list.reshape(1, -1)]
                add_list = np.array([row, col, n2, n3])
                arr = np.r_[arr, add_list.reshape(1, -1)]
                add_list = np.array([row, col, n1, n3])
                arr = np.r_[arr, add_list.reshape(1, -1)]

            # For tetra-cellular junction   With this resolution tetra-junction is the max.
            if len(neighbor) == 5:
                # print("(%d,%d)"%(row,col))
                # print(neighbor)
                n1 = neighbor[1]
                n2 = neighbor[2]
                n3 = neighbor[3]
                n4 = neighbor[4]

                add_list = np.array([row, col, n1, n2])
                arr = np.r_[arr, add_list.reshape(1, -1)]
                add_list = np.array([row, col, n2, n3])
                arr = np.r_[arr, add_list.reshape(1, -1)]
                add_list = np.array([row, col, n3, n4])
                arr = np.r_[arr, add_list.reshape(1, -1)]
                add_list = np.array([row, col, n1, n4])
                arr = np.r_[arr, add_list.reshape(1, -1)]
                add_list = np.array([row, col, n1, n3])
                arr = np.r_[arr, add_list.reshape(1, -1)]
                add_list = np.array([row, col, n2, n4])
                arr = np.r_[arr, add_list.reshape(1, -1)]

    bd_n_filename = "boundary_neighbor_cell_list_%d.txt" % frame
    np.savetxt(bddir_path + bd_n_filename, arr,
               header="row of boundary pixel, col of boundary pixel, neighbor_cell1, neighbor_cell2", fmt='%d')

    n_list = arr[:, 2:4]

    n_list_2 = list(map(list, set(map(tuple, n_list))))

    nlist_name = "neighbor_list_%d.txt" % frame
    np.savetxt(nlist_path + nlist_name, n_list_2,
               header="neighbor_cell1, neighbor_cell2", fmt='%d')

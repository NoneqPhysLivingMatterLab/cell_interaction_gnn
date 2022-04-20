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

"""Export Cell States for Cropped region"""

import tifffile
import matplotlib.pyplot as plt
from functions import system_utility as sutil
import numpy as np
from scipy import io
import os
import sys
version = 10
print("version=%d" % version)



figx = 3.14
figy = 3.14
left = 0.2
right = 0.9
bottom = 0.2
top = 0.8
dpi = 300

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

crop_shift_height = yaml_obj["crop_shift_height"]


# %%
#  Load mat file to numpy
base_path_file = "base_path.txt"
base_path = sutil.Read1LineText(base_path_file)
filename_txtfile = "segmentation_filename.txt"
filename = sutil.Read1LineText(filename_txtfile)
file_path = base_path + filename

if npy == 0:

    matdata = io.loadmat(file_path)
    data = matdata['segmat']

if npy == 1:
    matdata = np.load(file_path)
    data = matdata

total_frame = data.shape[0]
height = data.shape[1]
width = data.shape[2]


xmin = (width - crop_width)/2
xmax = width - 1 - (width - crop_width)/2
ymin = (height - crop_height)/2 + crop_shift_height
ymax = height - 1 - (height - crop_height)/2 + crop_shift_height


analdir_name = "analysis/"
analdir_path = base_path + analdir_name

cellID_nlist_dir_name = "nlist_cellID_w=%d_h=%d/" % (crop_width, crop_height)
cellID_nlist_dir_path = analdir_path + cellID_nlist_dir_name


# %%
# export cell state for Cropped  region
# cell type, NB:0, Del: 1, Div: 2

trackdata_alldir_name = "trackdata/"
trackdata_alldir_path = analdir_path + trackdata_alldir_name


trackdata_cropdir_name = "trackdata_crop_w=%d_h=%d/" % (
    crop_width, crop_height)
trackdata_cropdir_path = analdir_path + trackdata_cropdir_name

celltypedir_name = "celltype_w=%d_h=%d/" % (crop_width, crop_height)
celltypedir_path = analdir_path + celltypedir_name

sutil.MakeDirs(celltypedir_path)


Outsidedir_name = "Outside_celltype_w=%d_h=%d/" % (crop_width, crop_height)
Outsidedir_path = analdir_path + Outsidedir_name

sutil.MakeDirs(Outsidedir_path)

tripletdir_name = "triplet_w=%d_h=%d/" % (crop_width, crop_height)
tripletdir_path = analdir_path + tripletdir_name

sutil.MakeDirs(tripletdir_path)


triplet_parents = []
for frame_tmp in range(total_frame-1):

    filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame_tmp)
    current_data = np.loadtxt(trackdata_cropdir_path + filename)

    cellID_c = current_data[:, 3].astype(np.int64)
    parent_c = current_data[:, 2].astype(np.int64)

    cell_num = current_data.shape[0]

    filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame_tmp+1)
    next_data = np.loadtxt(trackdata_cropdir_path + filename)

    cellID_n = next_data[:, 3].astype(np.int64)
    parent_n = next_data[:, 2].astype(np.int64)

    filename = "trackdata_%d.txt" % (frame_tmp+1)
    next_data_all = np.loadtxt(trackdata_alldir_path + filename)

    cellID_n_all = next_data_all[:, 3].astype(np.int64)
    parent_n_all = next_data_all[:, 2].astype(np.int64)

    # cell type list is in the order of current cellID ,  1 is dying cell
    celltype = np.ones(cell_num)
    celltype = celltype.astype(np.int64)

    parent_n_list = parent_n.tolist()

    for i in cellID_n.tolist():

        celltype[np.where(cellID_c == i)] = 0  # nothing

    for i in parent_n_list:

        if len(np.where(cellID_c == i)[0]) > 0:

            if i != 0:  # Omit the parents ID = 0
                # Find the number of daughters from the all region
                num_daughters = len(np.where(parent_n_all == i)[0])

                if num_daughters > 2:
                    print("Error in the number of daughters")
                    print("ParentID=%d, num daughter=%d" % (i, num_daughters))
                    triplet_parents.append(i)
                    # print(num_daughters)

        celltype[np.where(cellID_c == i)] = 2   # dividing cell

    # Cells are tracked using the original images
    dif_list_tmp = np.where(celltype == 1)[0]

    Outside_NB = []
    Outside_Div = []
    for i in dif_list_tmp:

        if cellID_c[i] in cellID_n_all.tolist():
            # Cells which go outside the cropped region, but which is indeed NB if we look at the original image
            celltype[np.where(cellID_c == cellID_c[i])] = 0

            # print("Outside_NB")
            #print("CellID%d in frame %d"%(cellID_c[i],frame_tmp))
            Outside_NB.append(cellID_c[i])

        if cellID_c[i] in parent_n.tolist():
            num_daughters = len(np.where(parent_n == cellID_c[i])[0])
            # Cells which go outside the cropped region, but which is indeed Div if we look at the original image
            celltype[np.where(cellID_c == cellID_c[i])] = 2
            print("Outside_Div")
            print("CellID%d in frame %d" % (cellID_c[i], frame_tmp))
            Outside_Div.append(cellID_c[i])
            print("# of daughters = %d" % num_daughters)
            # If the daugther cells go outside the cropped region, check they are triplet.
            if num_daughters > 2:
                print("Error in the number of daughters")
                triplet_parents.append(cellID_c[i])

    data_save = np.zeros((cell_num, 2))
    data_save[:, 0] = cellID_c
    data_save[:, 1] = celltype

    celltype_filename = "celltype_%d.txt" % frame_tmp

    np.savetxt(celltypedir_path + celltype_filename,
               data_save, header="CellID, Celltype")

    Outside_NB_filename = "Outside_NB_%d.pickle" % frame_tmp
    sutil.PickleDump(Outside_NB, Outsidedir_path+Outside_NB_filename)

    Outside_Div_filename = "Outside_Div_%d.pickle" % frame_tmp
    sutil.PickleDump(Outside_Div, Outsidedir_path+Outside_Div_filename)


sutil.PickleDump(triplet_parents, tripletdir_path + "triplet.pickle")


print(triplet_parents)


# %%
# Semgemented images with nuclear position and network, only cropped region

print(triplet_parents)

tifdir_name = "tif/"
tifdir_path = analdir_path + tifdir_name


tif_network_dir_name = "tif_overlap_celltype_network_w=%d_h=%d/" % (
    crop_width, crop_height)
tif_network_dir_path = analdir_path + tif_network_dir_name

sutil.MakeDirs(tif_network_dir_path)


for frame in range(total_frame):

    filename = "trackdata_%d.txt" % (frame)
    trackdata = np.loadtxt(trackdata_alldir_path + filename)
    cellID_crop = trackdata[:, 3].astype(np.int64)
    parent_crop = trackdata[:, 2].astype(np.int64)
    x_crop = trackdata[:, 4]
    y_crop = trackdata[:, 5]

    if frame != (total_frame-1):
        celltype_filename = "celltype_%d.txt" % frame
        celltype = np.loadtxt(celltypedir_path + celltype_filename)

        Outside_NB_filename = "Outside_NB_%d.pickle" % frame
        Outside_NB = sutil.PickleLoad(Outsidedir_path+Outside_NB_filename)

        Outside_Div_filename = "Outside_Div_%d.pickle" % frame
        Outside_Div = sutil.PickleLoad(Outsidedir_path+Outside_Div_filename)

    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    filename = "nlist_cellID_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame)
    nlist = np.loadtxt(cellID_nlist_dir_path + filename)

    tifname = 'image_%d.tif' % frame
    img_np = tifffile.imread(tifdir_path + tifname)
    img_np[np.where(img_np > 0)] = 100

    cellID_list = nlist[:, 2].astype(np.int64)

    src_x = nlist[:, 4]
    src_y = nlist[:, 5]

    dst_x = nlist[:, 6]
    dst_y = nlist[:, 7]

    for npair in range(len(src_x)):

        ax.plot([src_x[npair], dst_x[npair]], [src_y[npair], dst_y[npair]], marker="o",
                ms=1, color="w", linewidth=0.5)  # inverse xy axis to overlap image

    if frame != (total_frame-1):
        for i, cellID in enumerate(celltype[:, 0].tolist()):
            celltype_tmp = celltype[i, 1].astype(np.int64)

            index = np.where(cellID_list == cellID)[0]

            x = src_x[index]
            y = src_y[index]

            if celltype_tmp == 0:
                ax.plot(x, y, marker="o", ms=1, c="green")

            if celltype_tmp == 1:
                ax.plot(x, y, marker="o", ms=1, c="blue")

            if celltype_tmp == 2:
                ax.plot(x, y, marker="o", ms=1, c="red")

    for cellID in triplet_parents:
        x = x_crop[np.where(cellID_crop == cellID)]
        y = y_crop[np.where(cellID_crop == cellID)]
        print(cellID)
        ax.plot(x, y, marker="s", ms=2, c="purple")

    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])

    ax.imshow(img_np.astype(np.int16))

    #ax.set_aspect('equal', adjustable='box')

    filename = 'tif_overlap_celltype_network_crop_w=%d_h=%d_%d.png' % (
        crop_width,  crop_height, frame)
    plt.savefig(tif_network_dir_path + filename, format="png", dpi=300)

    plt.close()


# %%
# export children

# cell type, NB:0, Del: 1, Div: 2

trackdata_cropdir_name = "trackdata_crop_w=%d_h=%d/" % (
    crop_width, crop_height)
trackdata_cropdir_path = analdir_path + trackdata_cropdir_name

childrendir_name = "children_w=%d_h=%d/" % (crop_width, crop_height)
childrendir_path = analdir_path + childrendir_name

sutil.MakeDirs(childrendir_path)

for frame_tmp in range(total_frame-1):
    print(frame_tmp)

    filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame_tmp)
    current_data = np.loadtxt(trackdata_cropdir_path + filename)

    cellID_c = current_data[:, 3].astype(np.int64)
    parent_c = current_data[:, 2].astype(np.int64)

    cell_num = current_data.shape[0]

    children_list = np.zeros((cell_num, 3)).astype(
        np.int64)  # CellID, child 1,  child2

    filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame_tmp+1)
    next_data = np.loadtxt(trackdata_cropdir_path + filename)

    cellID_n = next_data[:, 3].astype(np.int64)
    parent_n = next_data[:, 2].astype(np.int64)

    # cell type list is in the order of current cellID ,  1 is dying cell
    celltype = np.ones(cell_num)
    celltype = celltype.astype(np.int64)

    parent_n_list = parent_n.tolist()

    count = 0
    three_child = 0
    two_child = 0
    one_child = 0
    no_child = 0
    for i in cellID_c.tolist():

        children = cellID_n[np.where(parent_n == i)]

        if len(children) == 3:

            print(children)

            three_child += 1

        if len(children) == 2:
            children_list[count, 0] = i
            children_list[count, 1] = children[0]
            children_list[count, 2] = children[1]

            two_child += 1
        if len(children) == 1:
            children_list[count, 0] = i
            children_list[count, 1] = children[0]
            #children_list[count,2] = children[1]
            one_child += 1

        if len(children) == 0:
            children_list[count, 0] = i
            children_list[count, 1] = i
            #children_list[count,2] = children[1]
            no_child += 1

        count = count+1

    if cell_num == (two_child + one_child + no_child):
        print("tracing success")

    if cell_num != (two_child + one_child + no_child):

        print("tracing error")
        print(two_child)
        print(one_child)
        print(no_child)
        print(three_child)
        print(cell_num)

    childrenfile_name = "chidrenID_%d.txt" % frame_tmp
    np.savetxt(childrendir_path + childrenfile_name,
               children_list, header="CellID, ChildID1, ChildID2")


# %%
# export children-parent link


trackdata_cropdir_name = "trackdata_crop_w=%d_h=%d/" % (
    crop_width, crop_height)
trackdata_cropdir_path = analdir_path + trackdata_cropdir_name

parent_childrendir_name = "parent_children_w=%d_h=%d/" % (
    crop_width, crop_height)
parent_childrendir_path = analdir_path + parent_childrendir_name

sutil.MakeDirs(parent_childrendir_path)

for frame_tmp in range(total_frame-1):

    filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame_tmp)
    current_data = np.loadtxt(trackdata_cropdir_path + filename)

    cellID_c = current_data[:, 3].astype(np.int64)
    parent_c = current_data[:, 2].astype(np.int64)

    cell_num = current_data.shape[0]

    filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame_tmp+1)
    next_data = np.loadtxt(trackdata_cropdir_path + filename)

    cellID_n = next_data[:, 3].astype(np.int64)
    parent_n = next_data[:, 2].astype(np.int64)

    parent_n_list = parent_n.tolist()

    count = 0

    link_list = []

    for i in cellID_c.tolist():

        children = cellID_n[np.where(parent_n == i)]

        if len(children) == 2:

            link_list.append([i, children[0]])
            link_list.append([i, children[1]])
            count = count+1

        if len(children) == 1:
            link_list.append([i, children[0]])

            count = count+1

        if len(children) == 0:
            link_list.append([i, i])

            count = count+1

    link_list_nd = np.array(link_list).astype(np.int64)

    parent_children_filename = "parent_children_%d.txt" % frame_tmp
    np.savetxt(parent_childrendir_path + parent_children_filename,
               link_list_nd, header="CellID, ChildID")

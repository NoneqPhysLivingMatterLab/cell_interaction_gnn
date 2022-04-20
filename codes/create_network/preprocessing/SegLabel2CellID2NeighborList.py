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

"""Convert Segmentation label to Cell ID and Output NeighborList with Cell ID for cropped region."""

import tifffile
from functions import system_utility as sutil
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import os
import sys
DST_CONST = 100  # Large value with respect to the cell size
PIX_SEPARATE = 5


version = 10
print("version=%d" % version)



# parameters for plot
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 0.75
plt.rcParams['ytick.major.width'] = 0.75
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.75
# plt.locator_params(axis='x',nbins=20)
# plt.locator_params(axis='y',nbins=6)
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

label_ID_link_necessity = yaml_obj["label_ID_link_necessity"]

sim_switch = yaml_obj["sim_switch"]


error_correction_filename = "error_correction.pickle"
error_correction_list = sutil.PickleLoad(error_correction_filename)

print("error correction_list")
print(error_correction_list)


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


# %%
analdir_name = "analysis/"
analdir_path = base_path + analdir_name
nlist_name = "neighbor_list/"
nlist_path = analdir_path + nlist_name

trackdir_name = "trackdata/"
trackdir_path = analdir_path + trackdir_name

tifdir_name = "tif/"
tifdir_path = analdir_path + tifdir_name

tifoverlapdir_name = "tif_overlap/"
tifoverlapdir_path = analdir_path + tifoverlapdir_name

sutil.MakeDirs(tifoverlapdir_path)


# %%
# Semgemented images with nuclear position

for frame in range(total_frame):

    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    filename = "trackdata_%d.txt" % frame
    trackdata = np.loadtxt(trackdir_path+filename)

    nlist_name = "neighbor_list_%d.txt" % frame
    nlist = np.loadtxt(nlist_path + nlist_name)

    tifname = 'image_%d.tif' % frame
    img_np = tifffile.imread(tifdir_path + tifname)

    x = trackdata[:, 4]
    y = trackdata[:, 5]

    ax.scatter(x, y, s=5, c="w")  # inverse xy axis to overlap image
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    # plt.axes().set_aspect('equal')

    ax.imshow(img_np)

    tifoverlap_filename = 'image_overlap_%d.png' % frame
    plt.savefig(tifoverlapdir_path + tifoverlap_filename,
                format="png", dpi=300)

    plt.close()


# %%
# Semgemented images with nuclear position, only cropped region


tifoverlapcropdir_name = "tif_overlap_w=%d_h=%d/" % (crop_width, crop_height)
tifoverlapcropdir_path = analdir_path + tifoverlapcropdir_name

sutil.MakeDirs(tifoverlapcropdir_path)

trackdata_cropdir_name = "trackdata_crop_w=%d_h=%d/" % (
    crop_width, crop_height)
trackdata_cropdir_path = analdir_path + trackdata_cropdir_name

sutil.MakeDirs(trackdata_cropdir_path)


for frame in range(total_frame):

    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    filename = "trackdata_%d.txt" % frame
    trackdata = np.loadtxt(trackdir_path+filename)

    #print (trackdata)

    nlist_name = "neighbor_list_%d.txt" % frame
    nlist = np.loadtxt(nlist_path + nlist_name)
    #print (nlist)

    tifname = 'image_%d.tif' % frame
    img_np = tifffile.imread(tifdir_path + tifname)

    #img_np = np.loadtxt(tifdir_path + tifname)

    voronoi_area = trackdata[:, 6]
    voronoi_zero = np.where(voronoi_area < 0.001)  # To find voronoi_arrea = 0

    # Fill zero for the cells with zero voronoi area
    trackdata[voronoi_zero[0], 6] = 0.0
    trackdata[voronoi_zero[0], 7] = 0.0
    trackdata[voronoi_zero[0], 8] = 0.0
    trackdata[voronoi_zero[0], 9] = 0.0

    x = trackdata[:, 4]
    y = trackdata[:, 5]

    crop = np.where((x > xmin) & (x < xmax) & (y > ymin) & (y < ymax))

    # inverse xy axis to overlap image
    ax.scatter(x[crop], y[crop], s=5, c="w")
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    # plt.axes().set_aspect('equal')

    ax.imshow(img_np)

    tifoverlap_crop_filename = 'image_overlap_crop_w=%d_h=%d_%d.png' % (
        crop_width,  crop_height, frame)
    plt.savefig(tifoverlapcropdir_path +
                tifoverlap_crop_filename, format="png", dpi=300)

    plt.close()

    filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
        crop_width,  crop_height, frame)
    np.savetxt(trackdata_cropdir_path+filename,
               trackdata[crop[0], :], header="'Timeframe', 'CloneID', 'ParentID', 'CellID','PositionX','PositionY','VoronoiArea','G1MarkerInVoronoiArea','ActinSegmentationArea','G1MarkerInActinSegmentationArea' ")
    #print (trackdata)


# %%
# correspondence between cellID and label in the cropped region


cellID2label_crop_dir_name = "cellID2label_crop_w=%d_h=%d/" % (
    crop_width, crop_height)
cellID2label_crop_dir_path = analdir_path + cellID2label_crop_dir_name

sutil.MakeDirs(cellID2label_crop_dir_path)

mis_label_dirname = "miss_label/"
mis_label_dir_path = cellID2label_crop_dir_path + mis_label_dirname

sutil.MakeDirs(mis_label_dir_path)


for frame in range(total_frame):
    # for frame in [11]:
    print("\nframe=%d" % frame)

    filename = "trackdata_crop_w=%d_h=%d_%d.txt" % (
        crop_width,  crop_height, frame)
    trackdata_crop = np.loadtxt(trackdata_cropdir_path+filename)

    X_list_dbl = trackdata_crop[:, 4]
    Y_list_dbl = trackdata_crop[:, 5]

    cellID_list = trackdata_crop[:, 3].astype(np.int64)
    X_list = trackdata_crop[:, 4].astype(np.int64)
    Y_list = trackdata_crop[:, 5].astype(np.int64)

    ActinArea_list = trackdata_crop[:, 8]

    voronoi_area_list = trackdata_crop[:, 6].astype(np.int64)

    voronoi_zero_index = np.where(voronoi_area_list == 0)
    print("voronoi_zero_index = ")
    print(voronoi_zero_index[0])

    num_cell = len(cellID_list)

    tifname = 'image_%d.tif' % frame
    img_np = tifffile.imread(tifdir_path + tifname)

    nlist_name = "neighbor_list_%d.txt" % frame
    nlist = np.loadtxt(nlist_path + nlist_name)
    #print (nlist)

    array = np.zeros((num_cell, 5))

    # Correspond cell ID to the label number
    for i in range(num_cell):

        cellID = cellID_list[i]
        X = X_list[i]
        Y = Y_list[i]

        voronoi_aera = voronoi_area_list[i]

        if label_ID_link_necessity == 1:
            # X, Y is fiji coordinates. So we need to exchange X and Y
            cell_label = img_np[Y, X]
        if label_ID_link_necessity == 0:
            cell_label = cellID

        # for demo simulation data, we separate very close points by {PIX_SEPARATE} pix to avoid points on the segmentation border.
        if label_ID_link_necessity == 1 and sim_switch == 1:
            if cell_label == 0:
                X_dbl = X_list_dbl[i]
                Y_dbl = Y_list_dbl[i]

                dst = (X_list_dbl-X_dbl)**2 + (Y_list_dbl-Y_dbl)**2

                dst[i] = DST_CONST

                argmin = np.argmin(dst)

                print("argmin")
                print(argmin)

                dir_x = X_dbl-X_list_dbl[argmin]
                dir_y = Y_dbl-Y_list_dbl[argmin]
                norm = np.sqrt(dir_x**2 + dir_y**2)

                norm_x = dir_x/norm
                norm_y = dir_y/norm

                X_tmp = int(X_dbl + norm_x*PIX_SEPARATE)
                Y_tmp = int(Y_dbl + norm_y*PIX_SEPARATE)

                # X, Y is fiji coordinates. So we need to exchange X and Y
                cell_label = img_np[Y_tmp, X_tmp]

        array[i, :] = [cellID, cell_label, X, Y, voronoi_aera]

    array = array.astype(np.int64)

    array_correct = np.copy(array)

    array_cellID = array[:, 0]
    array_label = array[:, 1]
    array_X = array[:, 2]
    array_Y = array[:, 3]
    array_voronoi = array[:, 4]

    u, counts = np.unique(array_label, return_counts=True)
    multi_label = u[np.where(counts >= 2)[0]]

    print("multi_label is")
    print(multi_label)

    del_cellID = []

    """
     ############# Too small area size error  ##############
    print("Small area check") 
    small_area_list = array[np.where((ActinArea_list<10))]
    for small_area_cell in small_area_list:
        if len(small_area_cell)>0: 
            #print(small_area_cell)
            if small_area_cell[4]!=0: # if not voronoi area zero cell
                print(small_area_cell)

     ############# Too large area size error  ##############
    print("Large area check") 
    large_area_list = array[np.where((ActinArea_list>2000))]
    for large_area_cell in large_area_list:
        if len(large_area_cell)>0: 
            #print(small_area_cell)
            if large_area_cell[4]!=0: # if not voronoi area zero cell
                print(large_area_cell)
                
    """

    ############# if the point is on the boundary ###########

    label_tmp = 0

    if len(np.where(array_label == label_tmp)[0]) != 0:
        print("# of points on the boundary")
        print(len(np.where(array_label == label_tmp)[0]))
        cellID_error = array_cellID[np.where(array_label == label_tmp)]
        # print(array_X)
        X_error = array_X[np.where(array_label == label_tmp)]
        Y_error = array_Y[np.where(array_label == label_tmp)]

        vornoi_error = array_voronoi[np.where(array_label == label_tmp)]

        print("Error: Label, cellID, X, Y, Voronoi area")
        print([label_tmp, cellID_error, X_error, Y_error, vornoi_error])
        mis_label_list = np.array(
            [cellID_error, X_error, Y_error, vornoi_error])
        filename = "miss_frame=%d_label=%d.txt" % (frame, label_tmp)
        np.savetxt(mis_label_dir_path + filename, mis_label_list.T,
                   header="CellID, X, Y, voronoi")

        print("Error type is boundary")

        for k in range(len(cellID_error)):
            # print("cellID_error")
            print("cellID_error=%d" % cellID_error[k])

            #print("Empty label")

            if vornoi_error[k] != 0:

                candidate_list = np.array([]).astype(np.int64)
                for i in range(3):
                    for j in range(3):
                        Y_tmp = Y_error[k]+i-1
                        X_tmp = X_error[k]+j-1
                        #print([Y_tmp,X_tmp] )

                        candidate = img_np[Y_tmp, X_tmp].astype(np.int64)
                        if (candidate != 0) & (len(np.where(u == candidate)[0]) == 0):
                            # print(np.where(u==candidate)[0])
                            # print(len(np.where(u==candidate)[0]))
                            # print(candidate)
                            candidate_list = np.append(
                                candidate_list, candidate)

                if len(np.unique(candidate_list)) == 1:
                    print("Relabel success(border)")
                    print(candidate_list)

                    correct_label = candidate_list[0]

                    # correct array_label
                    array_correct[np.where(
                        array_cellID == cellID_error[k]), 1] = correct_label
                    #array_label =array[:,1]

                if len(np.unique(candidate_list)) != 1:
                    #print("Relabel error")

                    print('Relabel error(border)')

                    # print(cellID_error[k])
                    print(np.unique(candidate_list))

                    ########## Exceptions ##########

                    if label_ID_link_necessity == 1:
                        # print(error_correction_list)

                        if len(error_correction_list) > 0:

                            range(len(error_correction_list))
                            for error_cnt in range(len(error_correction_list)):

                                error = error_correction_list[error_cnt]
                                # print(error[0])
                                # print(error[1])
                                # print(frame)
                                # print(cellID_error[k])
                                if frame == error[0] and cellID_error[k] == error[1]:

                                    print("exception for cellID %d of the celllabel %d in frame %d" % (
                                        error[1], error[2], error[0]))

                                    array_correct[np.where(
                                        array_cellID == cellID_error[k]), 1] = error[2]

            else:
                print("div-dif cell on a border")
                print("cellID=%d" % cellID_error[k])
                del_cellID.append(cellID_error[k])

    ############# if the point is not on the boundary ###########
    ######## Identify Div and dif cell by looking at zero voronoi area #####

    print("\nFor errors not associated with border cells")

    for label_tmp in multi_label:

        if label_tmp != 0:  # if the point is mislabeled (double labeled)

            cellID_error = array_cellID[np.where(array_label == label_tmp)]
            # print(array_X)
            X_error = array_X[np.where(array_label == label_tmp)]
            Y_error = array_Y[np.where(array_label == label_tmp)]

            vornoi_error = array_voronoi[np.where(array_label == label_tmp)]

            print("Error: Label, cellID, X, Y, Voronoi area")
            print([label_tmp, cellID_error, X_error, Y_error, vornoi_error])
            mis_label_list = np.array(
                [cellID_error, X_error, Y_error, vornoi_error])
            filename = "miss_frame=%d_label=%d.txt" % (frame, label_tmp)
            np.savetxt(mis_label_dir_path + filename,
                       mis_label_list.T, header="CellID, X, Y, voronoi")

            if len(cellID_error) >= 3:
                print("Error! More than triple labeling")
            elif len(cellID_error) == 2:
                voronoi_zero_index = np.where(vornoi_error == 0)
                print(len(voronoi_zero_index[0]))

                if len(voronoi_zero_index[0]) == 1:
                    print("Doubling by Div and dif cell")
                    cellID_delete = cellID_error[voronoi_zero_index[0]]
                    print(cellID_delete[0])
                    del_cellID.append(cellID_delete[0])
                    # print(del_cellID)

                elif len(voronoi_zero_index[0]) == 0:

                    print('Real doubling label')

                else:
                    #print("other reason for double labelling")

                    print('other reason for double labelling')

    filename = "cellID2label_before_correction_w=%d_h=%d_%d.txt" % (
        crop_width,  crop_height, frame)
    np.savetxt(cellID2label_crop_dir_path + filename, array,
               header="CellID, CellLabel, X (fiji), Y (fiji), voronoi_area")

    print("cell ID to delete")
    print(del_cellID)

    # print(array_correct[64])
    # print(array_correct[65])
    for id_del in del_cellID:
        # print(np.where(array_correct[:,0]==id_del))
        array_correct = np.delete(array_correct, np.where(
            array_correct[:, 0] == id_del), 0)

    # print(array_correct[64])

    filename = "deleteCellID_w=%d_h=%d_%d.pickle" % (
        crop_width,  crop_height, frame)
    sutil.PickleDump(del_cellID, cellID2label_crop_dir_path + filename)

    filename = "cellID2label_after_correction_w=%d_h=%d_%d.txt" % (
        crop_width,  crop_height, frame)
    np.savetxt(cellID2label_crop_dir_path + filename, array_correct,
               header="CellID, CellLabel, X (fiji), Y (fiji), voronoi_area")


# %%
# Detect cells on the boundary

print("Check edge cell Errors")
# print(base_path)

for frame in range(total_frame):

    filename = "cellID2label_after_correction_w=%d_h=%d_%d.txt" % (
        crop_width,  crop_height, frame)

    array = np.loadtxt(cellID2label_crop_dir_path + filename).astype(np.int64)

    cellID_list = array[:, 0]
    cell_label_list = array[:, 1]
    X_array = array[:, 2]
    Y_array = array[:, 3]

    tifname = 'image_%d.tif' % frame
    img_np = tifffile.imread(tifdir_path + tifname)

    for i in range(len(cellID_list)):

        cellID = cellID_list[i]
        X = X_array[i]
        Y = Y_array[i]
        cell_label = cell_label_list[i]

        label_pix_list = np.where(img_np == cell_label)
        # print(label_pix_list[0])
        len1 = len(np.where(label_pix_list[0] == 0)[0])
        len2 = len(np.where(label_pix_list[0] == height-1)[0])
        len3 = len(np.where(label_pix_list[1] == 0)[0])
        len4 = len(np.where(label_pix_list[1] == width-1)[0])

        if (len1 != 0) or (len2 != 0) or (len3 != 0) or (len4 != 0):
            print("CAUTION: Edge cell error")
            print("frame =%d, CellID=%d, label=%d, X =%f, Y=%f, " %
                  (frame, cellID, cell_label, X, Y))


# %%
# convert neighbor_list into the neighbor_list by cellID

cellID_nlist_dir_name = "nlist_cellID_w=%d_h=%d/" % (crop_width, crop_height)
cellID_nlist_dir_path = analdir_path + cellID_nlist_dir_name

sutil.MakeDirs(cellID_nlist_dir_path)

for frame in range(total_frame):

    filename = "cellID2label_after_correction_w=%d_h=%d_%d.txt" % (
        crop_width,  crop_height, frame)

    array = np.loadtxt(cellID2label_crop_dir_path + filename).astype(np.int64)

    cellID = array[:, 0]
    cell_label = array[:, 1]
    X_array = array[:, 2]
    Y_array = array[:, 3]

    nlist_name = "neighbor_list_%d.txt" % frame
    nlist = np.loadtxt(nlist_path + nlist_name).astype(np.int64)

    nlist2 = nlist[:, [1, 0]]
    nlist_dual = np.concatenate([nlist, nlist2])

    result_arr = np.empty((0, 8), int)

    for neighbor in nlist_dual.tolist():

        src_label = neighbor[0]
        dst_label = neighbor[1]
        # if we have both labels in nlist  in cell_label
        if (src_label in cell_label) & (dst_label in cell_label):

            src_ID = cellID[np.where(cell_label == src_label)[0]]
            dst_ID = cellID[np.where(cell_label == dst_label)[0]]
            src_X = X_array[np.where(cell_label == src_label)[0]]
            src_Y = Y_array[np.where(cell_label == src_label)[0]]
            dst_X = X_array[np.where(cell_label == dst_label)[0]]
            dst_Y = Y_array[np.where(cell_label == dst_label)[0]]

            result = np.array([[src_label, dst_label, src_ID[0],
                              dst_ID[0], src_X[0], src_Y[0], dst_X[0], dst_Y[0]]])

            result_arr = np.append(result_arr, result, axis=0)

    filename = "nlist_cellID_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame)
    np.savetxt(cellID_nlist_dir_path + filename, result_arr,
               header="src_label, dst_label, src_ID, dst_ID, src_X, src_Y, dst_X, dst_Y")


# %%
# Semgemented images with nuclear position and network, only cropped region


tif_network_dir_name = "tif_overlap_network_w=%d_h=%d/" % (
    crop_width, crop_height)
tif_network_dir_path = analdir_path + tif_network_dir_name

sutil.MakeDirs(tif_network_dir_path)


for frame in range(total_frame):

    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    filename = "nlist_cellID_w=%d_h=%d_%d.txt" % (
        crop_width, crop_height, frame)
    nlist = np.loadtxt(cellID_nlist_dir_path + filename)

    tifname = 'image_%d.tif' % frame
    img_np = tifffile.imread(tifdir_path + tifname)

    src_x = nlist[:, 4]
    src_y = nlist[:, 5]

    dst_x = nlist[:, 6]
    dst_y = nlist[:, 7]

    for npair in range(len(src_x)):

        ax.plot([src_x[npair], dst_x[npair]], [src_y[npair], dst_y[npair]], marker="o",
                ms=1, color="w", linewidth=1)  # inverse xy axis to overlap image

    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    # plt.axes().set_aspect('equal')

    ax.imshow(img_np)

    filename = 'tif_overlap_network_crop_w=%d_h=%d_%d.png' % (
        crop_width,  crop_height, frame)
    plt.savefig(tif_network_dir_path + filename, format="png", dpi=300)

    plt.close()

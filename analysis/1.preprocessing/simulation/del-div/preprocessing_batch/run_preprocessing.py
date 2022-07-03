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

import subprocess
import yaml


from functions import system_utility as sutil

# Detect git repository path from system_utility.py.
sutil_file_path = sutil.__file__
remove_letter = "codes/functions/system_utility.py"
git_rep_base = sutil_file_path.replace(remove_letter, '')


program_path1 = git_rep_base + \
    "codes/create_network/preprocessing/DetectNeighborCells.py"
program_path2 = git_rep_base + "codes/create_network/preprocessing/ExtractCellType.py"
program_path3 = git_rep_base + \
    "codes/create_network/preprocessing/SegLabel2CellID2NeighborList.py"
program_path4 = git_rep_base + \
    "codes/create_network/preprocessing/ExportCellStateForCrop.py"


filename_yml = 'input_preprocessing_list.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

base_path_list = yaml_obj["base_path"]
segmentation_filename_list = yaml_obj["segmentation_filename"]
lineage_filename_list = yaml_obj["lineage_filename"]

crop_height_list = yaml_obj["crop_height"]
crop_width_list = yaml_obj["crop_width"]
crop_shift_height_list = yaml_obj["crop_shift_height"]
npy_list = yaml_obj["npy"]

label_ID_link_necessity_list = yaml_obj["label_ID_link_necessity"]

error_correction_list = yaml_obj["error_correction_list"]

sim_switch_list = yaml_obj["sim_switch"]
# -


for count, base_path in enumerate(base_path_list):
    print(base_path)

    base_w = "base_path.txt"
    mat_w = "segmentation_filename.txt"
    lineage_w = "lineage_filename.txt"

    with open(base_w, mode='w') as f:
        f.write(base_path_list[count])
    with open(mat_w, mode='w') as f2:
        f2.write(segmentation_filename_list[count])
    with open(lineage_w, mode='w') as f3:
        f3.write(lineage_filename_list[count])

    crop_height = crop_height_list[count]
    crop_width = crop_width_list[count]
    crop_shift_height = crop_shift_height_list[count]
    npy = npy_list[count]

    label_ID_link_necessity = label_ID_link_necessity_list[count]

    error_correction = error_correction_list[count]

    sim_switch = sim_switch_list[count]

    obj = {"crop_height": crop_height, "crop_width": crop_width, "crop_shift_height": crop_shift_height,
           "npy": npy, "label_ID_link_necessity": label_ID_link_necessity, "sim_switch": sim_switch}

    yaml_output = "input_preprocessing.yml"
    with open(yaml_output, 'w') as file:
        yaml.dump(obj, file)

    error_correction_filename = "error_correction.pickle"
    sutil.PickleDump(error_correction, error_correction_filename)

    with open(base_path_list[count] + "output_preprocessing_DetectNeighborCells_%d_%d.txt" % (crop_width, crop_height), 'w') as fp:

        proc = subprocess.run(['python', program_path1], stdout=fp, stderr=fp)

    with open(base_path_list[count] + "output_preprocessing_ExtractCellType_%d_%d.txt" % (crop_width, crop_height), 'w') as fp:

        proc = subprocess.run(['python', program_path2], stdout=fp, stderr=fp)

    with open(base_path_list[count] + "output_preprocessing_SegLabel2CellID2NeighborList_%d_%d.txt" % (crop_width, crop_height), 'w') as fp:

        proc = subprocess.run(['python', program_path3], stdout=fp, stderr=fp)

    with open(base_path_list[count] + "output_preprocessing_ExportCellStateForCrop_%d_%d.txt" % (crop_width, crop_height), 'w') as fp:

        proc = subprocess.run(['python', program_path4], stdout=fp, stderr=fp)



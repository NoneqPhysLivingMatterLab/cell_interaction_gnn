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
# import sys
import numpy as np


from functions import system_utility as sutil


# Detect git repository path from system_utility.py.
sutil_file_path = sutil.__file__
remove_letter = "codes/functions/system_utility.py"
git_rep_base = sutil_file_path.replace(remove_letter, '')

program_path3 = git_rep_base + \
    "codes/create_network/load_graph_add_G1perArea/load_graph_add_G1perArea.py"

base_list = sutil.ReadLinesText("base_path_list.txt")
crop_size_list = np.loadtxt(
    "crop_size_list.txt", delimiter=",").astype(np.int16)


# +

for count, base in enumerate(base_list):
    print(base)
    base_w = "base_path.txt"

    crop_size_w = "crop_size.txt"

    with open(base_w, mode='w') as f:
        f.write(base_list[count])

    print(crop_size_list[count])
    np.savetxt(crop_size_w, crop_size_list[count])

    crop_width = crop_size_list[count, 0]
    crop_height = crop_size_list[count, 1]
    crop_height_shift = crop_size_list[count, 2]

    with open(base_list[count] + "output_load_graph_add_G1perArea_%d_%d_%d.txt" % (crop_width, crop_height, crop_height_shift), 'w') as fp:
        proc = subprocess.run(["python", program_path3], stdout=fp, stderr=fp)

# -



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

from functions import system_utility as sutil
import subprocess
import os
import time


def Read1LineText(path):
    f = open(path)
    line = f.readlines()[0]
    f.close()
    return line


# Detect git repository path from system_utility.py.
sutil_file_path = sutil.__file__
remove_letter = "codes/functions/system_utility.py"
git_rep_base = sutil_file_path.replace(remove_letter, '')


# Set the path of the traing program
py_filepath = git_rep_base + "codes/training/run_training_cell_fate_gnn.py"

# set the number of GPU you want to use
gpu = 0
# set the number of samples you want to run
n_batch = 2


path_gpu = "./gpu.txt"
with open(path_gpu, 'w') as f:
    f.write("%s\n" % gpu)


path = os.getcwd()
# print(path)
# -

gpu_load = Read1LineText(path_gpu)
# print(gpu_load[0])
# print(os.path.isfile(path_gpu))

for i in range(n_batch):
    print(i)

    with open("./output_network_%d.txt" % i, 'w') as fp:
        proc = subprocess.Popen(['python', py_filepath], stdout=fp, stderr=fp)
        print("process id = %s" % proc.pid)
    time.sleep(20)

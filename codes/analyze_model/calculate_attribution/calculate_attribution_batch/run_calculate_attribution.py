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
import time
import os


from functions import system_utility as sutil

# Detect git repository path from system_utility.py.
sutil_file_path = sutil.__file__
remove_letter = "codes/functions/system_utility.py"
git_rep_base = sutil_file_path.replace(remove_letter, '')


program_path = git_rep_base + \
    "codes/analyze_model/calculate_attribution/calculate_attribution.py"
gpu = 1


path_gpu = "./gpu.txt"
with open(path_gpu, 'w') as f:
    f.write("%s\n" % gpu)


base_dir = sutil.ReadLinesText("base_path_list.txt")

base_list = sutil.RemoveCyberduckPrefixSuffix(base_dir)
print(base_list)

# -

for count, base_w in enumerate(base_list):
    print(base_w)

    base_w = "base_path.txt"

    with open(base_w, mode='w') as f:
        f.write(base_list[count])

    #msg = run_and_capture(['sh', 'all.sh'])
    #print (msg)
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d"%gpu

    with open(base_list[count] + "/output_calculate_attribution.txt", 'w') as fp:
        # Use this to run parallel. But to load filename correctly. we wait.
        proc = subprocess.Popen(['python', program_path], stdout=fp, stderr=fp)

        print("process id = %s" % proc.pid)

    time.sleep(60)

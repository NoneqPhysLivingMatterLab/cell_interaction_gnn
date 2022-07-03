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

gpu = 1
path_gpu = "./gpu.txt"
with open(path_gpu, 'w') as f:
    f.write("%s\n" % gpu)
    

# Detect git repository path from system_utility.py.
sutil_file_path = sutil.__file__
remove_letter = "codes/functions/system_utility.py"
git_rep_base = sutil_file_path.replace(remove_letter, '')


program_path1 = git_rep_base + \
    "codes/analyze_model/softmax_to_pool_attribution/plot_softmax_score.py"
program_path2 = git_rep_base + \
    "codes/analyze_model/softmax_to_pool_attribution/summarize_attribution.py"
program_path3 = git_rep_base + \
    "codes/analyze_model/softmax_to_pool_attribution/pool_attribution.py"


base_dir = sutil.ReadLinesText("base_path_list.txt")

base_list = sutil.RemoveCyberduckPrefixSuffix(base_dir)
print(base_list)

# -

for count, base_w in enumerate(base_list):
    print(base_w)

    base_w = "base_path.txt"

    with open(base_w, mode='w') as f:
        f.write(base_list[count])

    os.environ['CUDA_VISIBLE_DEVICES'] = "%d"%gpu
    
    time_sta = time.time()
    
    with open(base_list[count] + "/output_softmax_to_pool_attribution_plot_softmax_score.txt", 'w') as fp:
        # Use run to load filenames safely.
        proc = subprocess.run(['python', program_path1], stdout=fp, stderr=fp)
        print("finished program1")
        
    time_end = time.time()
    tim = (time_end- time_sta)/60
    print("%f min"%tim)
    time_sta = time.time()
    
    with open(base_list[count] + "/output_softmax_to_pool_attribution_summarize_attribution.txt", 'w') as fp:
        # Use run to load filenames safely.
        proc = subprocess.run(['python', program_path2], stdout=fp, stderr=fp)
        print("finished program2")
        
    time_end = time.time()
    tim = (time_end- time_sta)/60
    print("%f min"%tim)
    time_sta = time.time()

    with open(base_list[count] + "/output_softmax_to_pool_attribution_pool_attribution.txt", 'w') as fp:
        # Use run to load filenames safely.
        proc = subprocess.run(['python', program_path3], stdout=fp, stderr=fp)
        print("finished program3")
        #print( "process id = %s" % proc.pid )
    time_end = time.time()
    tim = (time_end- time_sta)/60
    print("%f min"%tim)



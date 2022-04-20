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
# import numpy as np


from functions import system_utility as sutil

# Detect git repository path from system_utility.py.
sutil_file_path = sutil.__file__
remove_letter = "codes/functions/system_utility.py"
git_rep_base = sutil_file_path.replace(remove_letter, '')


program_path1 = git_rep_base + \
    "codes/analyze_model/analyze_prediction_performance/make_sample_name_list.py"
program_path2 = git_rep_base + \
    "codes/analyze_model/analyze_prediction_performance/plot_performance_vs_time.py"
program_path3 = git_rep_base + \
    "codes/analyze_model/analyze_prediction_performance/plot_auc.py"

program_path4 = git_rep_base + \
    "codes/analyze_model/analyze_prediction_performance/average_confusion_matrix.py"
# -

with open("output_make_sample_name_list.txt", 'w') as fp:
    proc = subprocess.run(["python", program_path1], stdout=fp, stderr=fp)

with open("output_plot_performance_vs_time.txt", 'w') as fp:
    proc = subprocess.run(["python", program_path2], stdout=fp, stderr=fp)

with open("output_plot_auc.txt", 'w') as fp:
    proc = subprocess.run(["python", program_path3], stdout=fp, stderr=fp)

with open("output_confusion_matrix.txt", 'w') as fp:
    proc = subprocess.run(["python", program_path4], stdout=fp, stderr=fp)

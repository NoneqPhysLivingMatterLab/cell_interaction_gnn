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
# import time
# import os


from functions import system_utility as sutil

# Detect git repository path from system_utility.py.
sutil_file_path = sutil.__file__
remove_letter = "codes/functions/system_utility.py"
git_rep_base = sutil_file_path.replace(remove_letter, '')


program_path1 = git_rep_base + \
    "codes/analyze_model/bar_plot_attribution/sample_average_attribution.py"

program_path2 = git_rep_base + \
    "codes/analyze_model/bar_plot_attribution/bar_plot_attribution_sample_average.py"


# +

with open("../output_sample_average_attribution.txt", 'w') as fp:
    # Use run to load filenames safely.
    proc = subprocess.run(['python', program_path1], stdout=fp, stderr=fp)
    print("finished program1")

with open("../output_bar_plot_attribution_sample_average.txt", 'w') as fp:
    # Use run to load filenames safely.
    proc = subprocess.run(['python', program_path2], stdout=fp, stderr=fp)
    print("finished program2")

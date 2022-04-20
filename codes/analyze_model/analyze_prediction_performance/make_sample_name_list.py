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

"""Make sample filename list"""

from functions import system_utility as sutil
import os
import sys
version = 1
print("version=%d" % version)


filename_yml = 'input_analyze_prediction_performance.yml'
path_yml = "./" + filename_yml
# Load parameters
yaml_obj = sutil.LoadYml(path_yml)

base_name = yaml_obj["base_name"]
dir_base_list = yaml_obj["dir_base_list"]

index_list = yaml_obj["index_list"]

work_dir = yaml_obj["work_dir"]


for index in index_list:
    filename = base_name + "-sample%d" % index + ".txt"

    dir_base_list_tmp = []
    for dir_base in dir_base_list:
        dir_base_list_tmp.append(dir_base + "_%d" % index)

    sutil.SaveListText(dir_base_list_tmp, work_dir + filename)
# -

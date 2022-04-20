# -*- coding: utf-8 -*-
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


import os
import yaml
import pickle


def LoadYml(path):
    with open(path) as file:
        yaml_obj = yaml.safe_load(file)
    return yaml_obj


def MakeDirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def Read1LineText(path):
    f = open(path)
    line = f.readlines()[0]
    f.close()
    return line


def ReadLinesText(path):
    lists = []
    with open(path) as f:
        for line in f.read().splitlines():
            lists.append(line)
    return lists


def SaveListText(data_list, path):

    with open(path, 'w') as f:
        for list_element in data_list:
            f.write("%s\n" % list_element)


def PickleDump(obj, filepath):
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f)


def PickleLoad(filepath):
    with open(filepath, mode='rb') as f:
        data = pickle.load(f)
        return data


# %%
def RemoveCyberduckPrefixSuffix(name_list):

    prefix = "/work/takaki/homeostasis"
    prefix_new = "/mnt"

    suffix = ",,"

    name_list_new = []
    for name in name_list:

        name = name.replace(prefix, prefix_new)

        # print(name.find(suffix))
        if(name.find(suffix) != -1):
            name = name[:name.find(suffix)]

        name_list_new.append(name)

    return name_list_new

# +
# you can check the version of the required libraries

import sys
import torch
print(sys.version)

# +
import pprint

pprint.pprint(sys.path)
# -

print(torch.__version__)
print(torch.cuda.get_arch_list())
print(torch.version.cuda)

import dgl
print(dgl.__version__)

import numpy as np
print(np.__version__)

import matplotlib as mpl
print(mpl.__version__)

import sklearn 
print(sklearn.__version__)



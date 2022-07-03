# +
import subprocess
# import sys
import numpy as np


from functions import system_utility as sutil
# Detect git repository path from system_utility.py.
sutil_file_path = sutil.__file__
remove_letter = "codes/functions/system_utility.py"
git_rep_base = sutil_file_path.replace(remove_letter, '')


program_path3 = git_rep_base + \
    "codes/create_network/extract_subnetwork/extract_subnetwork.py"


base_list = sutil.ReadLinesText("base_path_list.txt")
lineage_file_list = sutil.ReadLinesText("lineage_filename_all.txt")
crop_size_list = np.loadtxt(
    "crop_size_list.txt", delimiter=",").astype(np.int16)

# +

for count, base in enumerate(base_list):
    print(base)
    base_w = "base_path.txt"

    lineage_w = "lineage_filename.txt"

    crop_size_w = "crop_size.txt"

    with open(base_w, mode='w') as f:
        f.write(base_list[count])

    with open(lineage_w, mode='w') as f2:
        f2.write(lineage_file_list[count])

    print(crop_size_list[count])
    np.savetxt(crop_size_w, crop_size_list[count])

    crop_width = crop_size_list[count, 0]
    crop_height = crop_size_list[count, 1]
    crop_height_shift = crop_size_list[count, 2]

    # with open(base_list[count]  + "output_create_network_%d_%d_%d.txt"%(crop_width,crop_height,crop_height_shift), 'w') as fp:
    #proc = subprocess.run(["python", program_path1],stdout=fp,stderr=fp)

    # with open(base_list[count]  + "output_read_network_edit_FL_%d_%d_%d.txt"%(crop_width,crop_height,crop_height_shift), 'w') as fp:
    #proc = subprocess.run(["python", program_path2],stdout=fp,stderr=fp)

    with open(base_list[count] + "output_extract_subnetwork_%d_%d_%d.txt" % (crop_width, crop_height, crop_height_shift), 'w') as fp:
        proc = subprocess.run(["python", program_path3], stdout=fp, stderr=fp)
# -



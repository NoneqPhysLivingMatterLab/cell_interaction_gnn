# Add features to graphs

- load_graph_add_G1perArea.py

Add G1 signal (G1 per area). Used only for paw.

## Usage: put load_graph_add_G1perArea somewhere. 

Run run_load_graph_add_G1perArea.py with base_path_list.txt, crop_size_list.txt,input_read_network_add_G1perArea.yml

## Detail
### load_graph_add_G1perArea.py

This program create, for example, "networknorm_all_1hot_rev-FLedit-G1Signal_crop_w=360_h=360". cell IDs at the final layer and the spatiotemporal graphs are output like create_network.py.

Output example: 

- networknorm_all_1hot_rev-FLedit-G1Signal_cellID_FinalLayer_noborder_num_w=360_h=360_time=4
- networknorm_all_1hot_rev-FLedit-G1Signal_num_w=360_h=360_time=4
- networknorm_all_1hot_rev-FLedit-G1Signal_trackdata_crop_w=360_h=360
- network_info



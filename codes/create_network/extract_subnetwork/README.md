# Extract subgraphs of each target cell for attribution analysis

- extract_subnetwork.py

Extract subnetworks for attribution for bidirectional(NSP) and unidirectional(SP) models. We output the list of the cell IDs of neighboring cells around each target cell, i.e. cells in the subnetwork of the target cell. 

## Usage

Put extract_subnetwork_batch in some directory. 

Run run_extract_subnetwork.py with .txt and .yml fiels to set the parameters.

This executes the code above sequentially for multiple samples.

## Details

### extract_subnetwork.py  
Input example: 
  - Graphs: 
  ex.
  ```
  "{directory including the lineage and segmentation data}/analysis/networknorm_all_1hot_rev_crop_w=320_h=320/networknorm_all_1hot_rev_num_w=320_h=320_time=4/NetworkWithFeartures_t=%dto%d.pickle"%(frame,frame+num_time-1)
  ```
 
  - List of the cell IDs in the final layer of each graph:  
  ex.
  ```
  "{directory including the lineage and segmentation data}/analysis/networknorm_all_1hot_rev_crop_w=320_h=320/networknorm_all_1hot_rev_cellID_FinalLayer_noborder_num_w=320_h=320_time=4/CellID_FinalLayer_t=%dto%d.txt"%(frame,frame+num_time-1)
  ```
    
We output the list of the cell IDs in the subnetwork of each target cell except the cells in the outmost layer. More specifically, the following lists defined in the code are output:  

  - AllTargetCellListNoDoubling: the list of the target and neighboring cell IDs, which are the cells in the subgraph of the target cell.   

 ```
 [[the target cell ID of the first layer(t0), neighbor1 ID(t0), neighbor2 ID(t0),... ],
 [the target cell ID of the 2nd layer(t1), neighbor1 ID(t1), neighbor2 ID(t1),...],
 ...]
 ```
 
The cell IDs are listed from the first layer. If the cell is absent in some frames, the list is blank. In the calculation of AllTargetCellList, some cell IDs are double counted, so we remove the double-counted cells and output it as AllTargetCellListNoDoubling. 

 - AllNeighborCellList is the the list of the neighboring cell IDs, which are the cells in the subgraph of the target cell. The cell IDs are listed from the first layer. If the cell is absent in some frames, the list is blank. 
 
  ```
 [[neighbor1 ID of the first layer(t0), neighbor2 ID of the first layer(t0),...],
 [neighbor1 ID of the 2nd layer(t1), neighbor2 ID of the 2nd layer(t0)(t1),...]
 ...]
 ```
 - LineageList: list of the cell IDs in the lineage of the target cell in the final layer. The cell IDs are listed from the final layer. 



AllTargetCellListNoDoubling, AllNeighborCellList, LineageList are saved for each cellID of the target cell as pickle files in the following directories: 
ex.
 - networknorm_all_1hot_rev_SP_AllNeighborCellList_noborder_num_w=320_h=320_time=4
 - networknorm_all_1hot_rev_SP_AllTargetCellListNoDoubling_noborder_num_w=320_h=320_time=4
 - networknorm_all_1hot_rev_SP_LineageList_noborder_num_w=320_h=320_time=4
 - networknorm_all_1hot_rev_NSP_AllNeighborCellList_noborder_num_w=320_h=320_time=4
 - networknorm_all_1hot_rev_NSP_AllTargetCellListNoDoubling_noborder_num_w=320_h=320_time=4
 - networknorm_all_1hot_rev_NSP_LineageList_noborder_num_w=320_h=320_time=4

In the directory names, SP and NSP means unidirectional and bidirectional models, respectively. 

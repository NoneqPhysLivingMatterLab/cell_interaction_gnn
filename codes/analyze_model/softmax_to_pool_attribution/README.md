# Plot softmax scores and pool attributions into feature categories

- plot_softmax_score.py: softmax scores with respect to \alpha, which is the parameter in IG calculation (see paper), is calculated. 
- summarize_attribution.py: only IGs of the cells in each subgraph are extracted from IntegratedGradient_list.pickle.
- pool_attribution.py: IGs are pooled for each feature types defined by T:Time frame number (0,1,2,,), P:Position(1:the target cell and 2:nearest neighbor cell), F:Index (0,1,2,,) of element of feature vector.

## Usage 

Put softmax_to_pool_attribution_batch somewhere, and then run run_softmax_to_pool_attribution.py with 

- base_path_list.txt: dir paths of samples
- input_plot_attribution.yml: Parameters

This executes three codes above sequentially.


## Details

### plot_softmax_score.py

Softmax scores with respect to $\alpha$, which is the variable of integral in IG calculation, are calculated. 
For example, "attribution_n=50_MaxMacroF1_nnet=11" is first created in each training sample directory, and the softmax scores are output in the following directories.

Output: 

- attribution_n=50_MaxMacroF1_nnet=11/test_summary/softmax/figs/   

    - SoftmaxOutput_p_all.png/pdf : scores for all the cells are plotted against $\alpha$. 
    - SoftmaxOutput_p_correct.png/pdf : scores only for correct prediction are plotted against $\alpha$.
    - SoftmaxOutput_p_wrong.png/pdf : scores only for wrong prediction are plotted against $\alpha$.
    

- attribution_n=50_MaxMacroF1_nnet=11/test_summary/softmax/data/   

    The corresponding numerical values are saved with the corresponding filenames such as `SoftmaxOutput_p_Delamination_all_y.pickle`. 



### summarize_attribution.py


The results are output in "attribution_n=50_MaxMacroF1" created by calculate_attribution.py. 
In IntegratedGradient_list.pickle, the IGs are calculated for all the cells in each input graph. In this program, we extract only IGs of the cells in each subgraph.

Input:

- attribution_n=50_MaxMacroF1/test/test_0/AllCells/0/cellID=62_reallabel=2/

    - AllTargetCellListNoDoubling.pickle:  List of cell IDs in each subgraph, listed for each time frame. See `codes/create_network/extract_subnetwork/README.md`
    - AllNeighborCellList.pickle:  List of only neighbor cell IDs in each subgraph, listed for each time frame. See `codes/create_network/extract_subnetwork/README.md`
    - IntegratedGradient_list.pickle: IGs of all the feature elements of all the cells in each input graph for each label.
    - LineageList.pickle: List of cell IDs of the target cell, listed from the final frame.

Output: 

- attribution_n=50_MaxMacroF1/test/test_0/AllCells/0/cellID=62_reallabel=2

    Only IGs of the cells in each subgraph are extracted and saved for each label. 

    - time_index_all.pickle: The time index (0,1,2,) is saved for the cell IDs listed in AllTargetCellListNoDoubling.pickle. Same format as AllTargetCellListNoDoubling.

    - position_all_path.pickle: The position index (1,2) is saved for the cell IDs listed in AllTargetCellListNoDoubling.pickle. Same format as AllTargetCellListNoDoublposition. Here, the position indeces are definced as 1:the target cell and 2:nearest neighbor cell.  

    - IntegratedGradient_target_1D.pickle: 1D array of IG of cells in a subgraph. ID list formatted with row:ID and column:feature element is flattend to 1D. The order of cells are the 1D-flattened version of AllTargetCellListNoDoubling.pickle.
 
        ```
        format: ["IG of feature1 of cell1 of in the subgraph","IG of feature2 of cell1 of in the subgraph",,,"IG of feature1 of cell2 of in the subgraph",,,]

        ```

    - features_target_1D.pickle: 1D array of feature values of cells in a subgraph. ID list formatted with row:ID and column:feature element is flattend to 1D. The order of cells are the 1D-flattened version of AllTargetCellListNoDoubling.pickle.
 
        ```
        format: ["feature1 of cell1 of in the subgraph","feature2 of cell1 of in the subgraph",,,"feature1 of cell2 of in the subgraph",,,]

        ```

    Run check_data_format.ipynb to check the data format. 

    - misc
    
        Sorted IGs are calculated but we don't use this for the further analysis. So you can ignore these data. 

        - attribution_sort_value_list.pickle
        - attribution_sort_1Dindex_list.pickle
        - attribution_sort_feature_index_list.pickle
        - attribution_sort_ID_index_list.pickle

        Absolute values of IG are calculated but we don't use this for the further analysis. So you can ignore these data. 

        - abs_attribution_sort_1Dindex_list.pickle
        - abs_attribution_sort_feature_index_list.pickle
        - abs_attribution_sort_ID_index_list.pickle
        - abs_attribution_sort_value_list.pickle
        - abs_IntegratedGradient_target_1D.pickle



### pool_attribution.py

IGs are pooled for each feature types defined by T:Time frame number (0,1,2,,), P:Position(1:the target cell and 2:nearest neighbor cell), F:Index (0,1,2,,) of element of feature vector. For example, T1P1F4 means the forth element of the feature vector of the target cell in the 1st time frame:t1. 


Input: 

- attribution_n=50_MaxMacroF1/test/test_0/AllCells/0/cellID=62_reallabel=2

    - time_index_all.pickle: see above. Output of summarize_attribution.py
    - position_all_path.pickle: see above. Output of summarize_attribution.py
    - IntegratedGradient_target_1D.pickle: see above. Output of summarize_attribution.py
    - features_target_1D.pickle: see above. Output of summarize_attribution.py

Output: 

- attribution_n=50_MaxMacroF1/test/test_0/AllCells/0/cellID=62_reallabel=2

    - cell_type_name_list.pickle: list of cell type name ("T%dP%dF%d"%(time_type,position_type,feature_type)) in the same order as IGs. 
    
        Using this cell type name list, we pool IGs in each subnetwork. 

    - ForEachCellType/data/T1P1F4

        IGs are pooled for each subgraph and for each feature category(ex.T1P1F4).

        - features.pickle
        - IG.pickle
        - index.pickle
        
- attribution_n=50_MaxMacroF1_nnet=11/test_summary/AllCells/data/0,1,2

    Pooled IG and feature values of all the subgraphs are aggregated from the above data for each subgraph.

    - 0/T1P1F2/
        - features_list_all.txt
        - IG_list_all.txt

    We use this data when we plot the pooled IGs.

    See check_data_format.ipynb to check the data format. 
            



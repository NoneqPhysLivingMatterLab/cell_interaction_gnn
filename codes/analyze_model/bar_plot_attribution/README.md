# Average the attribution over samples by pooling the features

- sample_average_attribution.py

    Average attributions with respect to each feature for each sample, and then sample-average, and save the result. 
    
- bar_plot_attribution_sample_average.py
    
    Bar plot the sample averaged IGs. 

## Usage

Put "bar_plot_attribution_batch" in some directory, and run run_bar_plot_attribution.py with base_path_list.txt and input_bar_plot_attribution.yml. 

This executes two codes above sequentially.


## Details 

### sample_average_attribution.py. 

IGs averaged over samples for each label are output.

Input: 

- attribution_n=50_MaxMacroF1_nnet=11/test_summary/AllCells/data/0,1,2   

    Pooled IG and feature values for all the subgraphs in a trained sample. 
    - 0/T1P1F2/
        - features_list_all.txt
        - IG_list_all.txt

Ouput: 

bar_plot_result/4time_ext_mean_ZZFR_sample=6/MeanForEachFutureFate/MaxMacroF1/AllCells/Raw/data,fig are created, and then saved the results in the following directories. 

- data 

    In this directory, we output the IGs averaged over samples for each label. The IGs are sorted as the feature name list in "label=%d_featurename_list.pickle". Labels are denoted by 0:NB,1:Del and 2:Div.

    - label={0,1,2}_average_mean.pickle: Sample-average of IGs for each label
    - label={0,1,2}_featurename_list.pickle: Names of features. 
    - label={0,1,2}_average_se.pickle:  Standard error in sample-averaging IGs for each label
    - label={0,1,2}_cellname_list.pickle: cell type format is "T%dP%dF%d"%(j,k,l)
    - label={0,1,2}_cellname2_list.pickle: cell type format is "T%d\nP%d\nF%d"%(j,k,l)

    
### bar_plot_attribution_sample_average.py


Sample-averaged IGs created by sample_average_attribution_to_bar_plot.py are plotted. 

Input: 

bar_plot_result/4time_ext_mean_ZZFR_sample=6/MeanForEachFutureFate/MaxMacroF1/AllCells/Raw/data/

- "label={0,1,2}_average_mean.pickle": Sample-average of IGs for each label
- "label={0,1,2}_featurename_list.pickle": Names of features. ex.'Div of Neighbor cell at t=-1'
- "label={0,1,2}_average_se.pickle":  Standard error in sample-averaging IGs for each label

        
Output: 

- bar_plot_result/4time_ext_mean_ZZFR_sample=6/MeanForEachFutureFate/MaxMacroF1/AllCells/Raw/fig

    Bar plots are output with various formats.

    - 4time_ext_mean_ZZFR_SampleAverage_IG_all_sample=6_standadized_woxlabel_group_nolegend_ylim_-0.20_0.20.png/pdf
    - 4time_ext_mean_ZZFR_3_SampleAverage_IG_all_sample=6_standadized_group_nolegend_ylim_-0.20_0.20_rotate.png
    - figure with other formats
        
        
- bar_plot_result/4time_ext_mean_ZZFR_sample=6/MeanForEachFutureFate/MaxMacroF1/AllCells/Raw/data/bar_plot
       
    - IGs_label={NB,Del,Div}.txt: Raw data of the bar plots of IG.
    - list_sample_averaged_IG_random_features_label={NB,Del,Div}.txt: sample average and SE of IG for random features.
    - standard_label={NB,Del,Div}.txt: min and max of the zone indicating the baseline IG level.
    - misc: pickled IG values, feature category names, SE, etc.
# Analyze the prediction peformance of the trained model

- make_sample_name_list.py: make text files with the list of directory paths for the analysis.

- plot_performance_vs_time.py: plot predictive performance such as AUC against epoch.

- plot_auc.py: plot the predictive performance (AUC) for the best model during the training time course, and plot the averaged peformance over samples. 


## Usage

Put run_analyze_prediction_performance in a directory, and run run_analyze_prediction_performance.py with .yml fiels to set the parameters.

This executes three codes above sequentially.

The summary directory is created as follows:

```
{work_dir}/summary_Performance-{base_name}-0.50

```
The results are output under the above directory. 

## Details

The summary directory is created as, for example,

```
{work_dir}/summary_Performance-{base_name}-0.50

```
The following results are output under the above directory. 


### make_sample_name_list.py


Input: input_analyze_prediction_performance.yml

Output: Text files with the list of the filenames which we analyze.

For example,
    - 4time_ext_mean_ZZFR-sample0.txt


### plot_performance_vs_time.py

Plot time evolution of predictive performance measures as listed below. 

Input: input_analyze_prediction_performance.yml and input_fig_config.yml

Output:  
- Recall_BalancedACC_time_evolution: Time evolution of recall and balanced ACC
- Recall_MacroF1Score_time_evolution: Time evolution of recall and macro-F1 score
- Precision_MacroF1Score_time_evolution: Time evolution of precision and macro-F1 score
- AUC_time_evolution: Time evolution of AUC


### plot_auc.py

Plot balanced ACC, macro-F1 score and AUCs averaged over samples. 
Input: input_analyze_prediction_performance.yml and input_fig_config.yml

Output:  
- MaxF1Score: Max macro F1 score averaged over samples. 
- MaxBalancedACC: Max balanced ACC averaged over samples. 
- AUCForMaxMacroF1Model: AUC of the max macro F1 score model averaged over samples. 
ex.
```
AUCForMaxMacroF1Model/all/4time_ext_mean_ZZFR/figs/AUCForMaxMacroF1Model.png
```
- AUCForMaxAUCModel: AUC of the max AUC model averaged over samples. 
- AUCForMaxACCModel: AUC of the max balanced ACC model averaged over samples. 

### average_confusion_matrix.py

Average confusion matrix for Max macro-F1 score models. 
The data is output in `ConfusionMatrixForMaxF1Model/{name_list[i]}/data`.

Output:  
- confusion_matrix_training_all_average.txt: average of confusion matrix of training data averaged over samples
- confusion_matrix_test_all_average.txt:  average of cconfusion matrix of text data averaged over samples
- confusion_matrix_training_all_std.txt:  standard deviation of confusion matrix of training data averaged over samples
- confusion_matrix_text_all_std.txt:  standard deviation of confusion matrix of training data averaged over samples
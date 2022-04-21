# Probing the rules of cell coordination in live tissues by interpretable machine learning based on graph neural networks

Codes used in the following paper:

Takaki Yamamoto, Katie Cockburn, Valentina Greco, Kyogo Kawaguchi

"Probing the rules of cell coordination in live tissues by interpretable machine learning based on graph neural networks"

[bioRxiv, 2021.06.23.449559 (2021).](https://www.biorxiv.org/content/10.1101/2021.06.23.449559v2)


# Abstract

Robustness in developing and homeostatic tissues is supported by various types of spatiotemporal cell-to-cell interactions. Although live imaging and cell tracking are powerful in providing direct evidence of cell coordination rules, extracting and comparing these rules across many tissues with potentially different length and timescales of coordination requires a versatile framework of analysis. Here we demonstrate that graph neural network (GNN) models are suited for this purpose, by showing how they can be applied to predict cell fate in tissues and utilized to infer the cell interactions governing the multicellular dynamics. Analyzing the live mammalian epidermis data, where spatiotemporal graphs constructed from cell tracks and cell contacts are given as inputs, GNN discovers distinct neighbor cell fate coordination rules that depend on the region of the body. This approach demonstrates how the GNN framework is powerful in inferring general cell interaction rules from live data without prior knowledge of the signaling involved. 

# Requirement

- pytorch 1.6.0
- python 3.7.7
- DGL (Deep Graph Library) 0.5.0
- numpy 1.18.5
- matplotlib 3.3.1
- sklern  0.23.2
- Docker

We used a Docker container installing by `docker pull pytorch:1.6.0-cuda10.1-cudnn7-runtime`


# Dataset

We convert the segmentation image from the live-image/simulation data and the track data into the spatio-temporal graphs (DGL format). 

## Data format

### Segmentation image (segmentation.npy)

Numpy array [T,H,W] of a image sequence, i.e. label image sequence is save as numpy array.
In the label images, the cell borders (width = 1 pix) are set to zero value. 

See `data/check_input_data_format.ipynb` to check the format. 

### Track data (lineage.npy)

Numpy array of 10 columns with the following fields.

['Timeframe', 'CloneID', 'ParentID', 'CellID','PositionX','PositionY','VoronoiArea','G1MarkerInVoronoiArea','ActinSegmentationArea','G1MarkerInActinSegmentationArea']

In 'Timeframe', the initial frame number is set to "0". 
In this paper, we used 'ActinSegmentationArea' (cell area) and 'G1MarkerInActinSegmentationArea' (Total G1 marker intensity per cell) only in paw epidermis. Also, we did not use the features 'VoronoiArea' and 'G1MarkerInVoronoiArea'. 

See `data/check_input_data_format.ipynb` to check the format. 

### Spatio-temporal graphs (DGL format) created from segmentation and track data

We used deep graph library (DGL) to make graph objects, and saved them as .pickle. 

The node type is defined by the frame number of each cell in each graph. We name the node names as  “t0”, “t1", “t2”... from the earliest frame (see figure below).
There are three edge types:  “time”, “time_rev”, “interaction”. “time” is the temporal edge, representing lineage, directing from the past to future, while “time_rev” is the temporal edge directing from the future to the past. “interaction” edges are spatial edges between neighboring cells. Also, you need to define cell features as node features, ex. “celltype_future_onehot2"(NFB in our manuscript), “random_feature”, etc. Regarding NFB, since NFB at the final time frame is the cell fates we try to predict, we set NFB at the final layer by the null vectors.

 - Node type
 
't0', 't1', 't2'...: ex. 't0' means the cell type of the cells in the first time frame, 't1' means the cell type of the cells in the second time frame. 

 - Edge type
    - Temporal
        - "time" : edges in normal direction
        - "time_rev" : edges in time-reversal direction
    - Spatial
        - "interaction"
        
<div align="center">        
<img src="codes/create_network/create_network/definition_graph.png" width="600">
</div>   


## Data used in the demo codes

In the demo codes, we apply the GNN model on the simulation data of tissue dynamics with a delamination-induced division rule and the experimental data of the mouse paw epidermis.

You can find the segmentation and track data in the following directories: 
```
   data/simulation/del-div/sample0 (training), sample1 (test)
   data/epidermis/paw/W-R1 (training), W-R2 (test)
```

# Quick demonstration for training of the GNN models

If you would like to quickly try to train the GNN model with sample graphs, please move to the following directory and run the sample codes with sample data.

```
codes/training_quick_demo
```

If you would like to create spatio-temporal graphs, train the GNN model and then calculate the attribution, see the usage below. 

# Usage

## 0. Pull this repository and export the path

Pull this repository, and then export the path for `{this repository}/codes` to import the libraries on python. For example, you can do this by creating .pth files as `/opt/conda/lib/python3.7/site-packages/.pth`, which is an example path for docker users, and then write the paths to `{this repository}/codes`.


## 1. Create spatio-temporal graphs (dgl format) from track data

We here explain how to create spatio-temporal graphs (dgl format) from track data. Hereafter, we show the usage of the pipline by applying it to the simulation data.

- Preprocessing

Run run_preprocessing.py in the following directory:
```
analysis/1.preprocessing/simulation/del-div/preprocessing_batch
```

- Create spatio-temporal graphs and output cell IDs (hereafter, we call "target IDs"), which is the list of node IDs in the final frame of each graph which you want to predict the fate of.

    Run run_create_network.py in the following directory:
    ```
    analysis/2.create_network/simulation/del-div/create_network_batch.
    ```
    We use the following files in the training. 

    - graphs
    
        For example, the graphs are output as the following file: t=0to3 in the filename means the graph created from frame 0 to 3. 
        
    ```
   data/simulation/del-div/analysis/networknorm_all_1hot_rev-FLedit_crop_w=320_h=320/networknorm_all_1hot_rev-FLedit_num_w=320_h=320_time=4/NetworkWithFeartures_t=0to3.pickle
   ```
    
     - IDs
     
        For example, the list of node IDs in the final frame of each graph which you want to predict the fate of is saved as the following file: t=0to3 in the filename means the graph created from frame 0 to 3. 
        
    ``` 
    data/simulation/del-div/analysis/networknorm_all_1hot_rev-FLedit_crop_w=320_h=320/networknorm_all_1hot_rev-FLedit_cellID_FinalLayer_noborder_num_w=320_h=320_time=4/CellID_FinalLayer_t=0to3.txt
    ```


- Extract subgraph information, which is necessary to calculate attribution 

Run run_extract_subnetwork.py in the following directory:

```
analysis/2.create_network/simulation/del-div/extract_subnetwork_batch
```

## 2. Train the GNN model

Run run_training_parallel.py in the following directory:

```
analysis/3.training_attribution/simulation/del-div/feature_ZZFR 
```

You can execute training of a GNN model (4-time cell external model with mean aggregation with NFB and randome features) using the simulation data as an example. 
Hereafter, the training will be performed for 6 samples on GPU in this demo codes (~1.1GB GPU memory is used for sample.).
You can change the number of samples by changing the number "n_batch" of samples in run_training_parallel.py.
Also, you can change any training parameteres in input_run_training_cell_fate_gnn.yml.



## 3. Analyze the peformance of the training

Run run_analyze_prediction_performance.py in the following directory:

```
analysis/3.training_attribution/simulation/del-div/feature_ZZFR/run_analyze_prediction_performance
```

You can obtain training curves and peformance averaged over samples in the following directory:

```
analysis/3.training_attribution/simulation/del-div/summary_Performance-4time_ext_mean_ZZFR-0.50
```

## 4. Calculate the attribution

- Run run_calculate_attribution.py in the following directory:

```
analysis/3.training_attribution/simulation/del-div/feature_ZZFR/calculate_attribution_batch
```

You can calculate attribution of the GNN model. 

- Run run_softmax_to_pool_attribution.py in the following directory: 

```
analysis/3.training_attribution/simulation/del-div/feature_ZZFR/softmax_to_pool_attribution_batch
```

You can pool attributions into each feature type defined by time frame, relative position and index of feature vector. 


## 5. Plot the attribution 



- Run run_bar_plot_attribution.py in the following directory: 

```
analysis/3.training_attribution/simulation/del-div/feature_ZZFR/bar_plot_attribution_batch
```

You can obtain the bar plots of the attribution in the following directory: 

```
analysis/3.training_attribution/simulation/del-div_for_share/ZZFR_2000/bar_plot_result/feature_ZZFR_sample=6/MeanForEachFutureFate/MaxMacroF1/AllCells/Raw/fig/feature_ZZFR_SampleAverage_IG_all_sample=6_standadized_group_nolegend_ylim_-0.20_0.20_rotate.png
```

as we show in Fig.4D of the paper.



# Reference

If this repository is helpful for your research, please cite the following publication:

Takaki Yamamoto, Katie Cockburn, Valentina Greco, Kyogo Kawaguchi

"Probing the rules of cell coordination in live tissues by interpretable machine learning based on graph neural networks"

[bioRxiv, 2021.06.23.449559 (2021).](https://www.biorxiv.org/content/10.1101/2021.06.23.449559v2)

# License

All the codes in this repository is licensed under the Apache License.

``` python
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
```
# Codes for training the GNN model

## Usage

Run run_training_parallel.py with .txt and .yml fiels to set the parameters.

In run_training_parallel.py, we need to set GPU number and the number of samples. 
The training code runs in parallel for the number of samples.


## Parameters for the training

You can set the parameters for the training in input_run_training_cell_fate_gnn.yml. 

## Features used in the training

You can select which features you use in the training. 

- 'Area_norm': normalized area.

- 'G1Signal_norm': normalized G1 signal.

- 'celltype_future_onehot2': next frame behavior (NFB) represented by the one-hot vector.

[1,0,0]: NB (No behavior), [0,1,0]: Del (Delamination), [0,0,1]: Div (Division)

If we want to use the correspoidng zero onehot vector, set 'zero_celltype_future_onehot2'. 

- 'random_feature'

Random number is generated upon training to assign random features.

- 'zero'

Zero values are set. 



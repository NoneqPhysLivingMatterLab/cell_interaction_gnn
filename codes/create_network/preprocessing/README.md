# Preprocessing before creating spatiotemporal graphs

- DetectNeighborCells.py: detect nighbor cells from segmenation .npy file.
- ExtractCellType.py: extract cell types (del,div,NB) from lineage .npy file for the whole area.
- SegLabel2CellID2NeighborList.py: correspond the segmentation label to cell ID, and output neighbor cell list by cell ID for the cropped region. 
- ExportCellStateForCrop.py: extract cell types (del,div,NB) for the cropped region.

In the following codes, we load the standard-format lineage and segmentation files.

## Usage

Put preprocessing_batch in a directory, and then run run_preprocessing.py with .yml fiels to set the parameters.

This executes the codes above sequentially for multiple samples.

## Details

### DetectNeighborCells.py

Detect nighbor cells from segmenation .npy file. The neighbor cell relations are identified from the label image (segmenation .npy file) including cell borders whose pixel values are set to zero and whose width is 1 pix. 

### ExtractCellType.py

Extract cell types (del,div,NB) from lineage .npy file for the whole area.

### SegLabel2CellID2NeighborList.py

Correspond the segmentation label to cell ID, and output neighbor cell list by cell ID for the cropped region. 

If you need manual error corrections, set the correct labels by defining `[frame, cellID_error, label]`.

```yml
error_correction_list: # if we need correction, input list [frame, cellID with labeling error, correct_label]
    - 
      - [13,1091,64]

    - 0

``` 
### ExportCellStateForCrop.py

Extract cell types (del,div,NB) for the cropped region. 
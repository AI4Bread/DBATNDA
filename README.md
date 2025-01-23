# DBATNDA
Dual Balanced Augmented Topological Non-coding RNA Disease Association in Heterogeneous Graphs

![DBATNDA](/image/figure1.jpg)

## Introduction
In this work, we propose the Dual Balanced Augmented Topological Non-coding RNA Disease Association (DBATNDA) model. DBATNDA constructs an Interaction Dual Graph (IDG) with LDAs, MDAs, and LMIs as nodes and introduces an efficient graph-based balanced topological augmentation mechanism to enhance node structural representation and adaptability to imbalanced data. This innovative approach enables fast and accurate predictions of ncRNA-disease associations through node classification view. To the best of our knowledge, no existing method employs such a dual-representation strategy to provide differentiated predictions for diverse ncRNAs and their disease associations while also enhance target specificity. Experimental results demonstrate DBATNDA’s superior performance compared to state-of-the-art models, while case studies confirm its practical significance in ncRNA-disease association prediction.

## RUN DBATNDA
### Requirements
The experiments are conducted in the following environment:
`Python 3.9.19` `PyTorch 2.0.0` `CUDA 11.8` `Numpy 1.26.3` `Pandas 1.4.4` `matplotlib 3.7.3` `sklearn 1.2.2` `seaborn 0.11.0` `torch-geometric 2.5.3`

### Data Preparation
The original data need to be preprocessed for the following work, if you want to use your own dataset, run `autoencoder.py` first to get the following data:
```
--lnc_fun.npy
--dis_fun.npy
--mi_fun.npy
--lnc_dis.npy
--dis_lnc.npy
--mi_dis.npy
--dis_mi.npy
--mi_lnc.npy
--lnc_mi.npy
```

### Graph Conversion
After preparing the data, you need to convert the data structure, run `graph_conversion.py` to get:
```
--node_features.pt
--edge_index.pt
--labels.pt
--labels_2.pt
--train_mask.pt
--test_mask.pt
```

### Prediction
run `main.py` to get the predicting results



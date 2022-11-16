<p align="center">
<img src="https://github.com/pievos101/Ensemble-GNN/blob/main/logo2.png" width="400">
</p>

# Ensemble-GNN: ensemble learning with graph neural networks for disease module discovery

## Installation

The GNN-Subnet package is required 


To install GNNSubNet run:

```python
pip install torch==1.11.0 
pip install torchvision==0.12.0
pip install torch-geometric==2.0.4
pip install torch-scatter==2.0.9
pip install torch-sparse==0.6.13

pip install GNNSubNet
```
Preferred versions are: torch==1.11.0, torchvision==0.12.0, torch-geometric==2.0.4, torch-scatter==2.0.9, and torch-sparse==0.6.13.

To install Ensemble-GNN from source code run:

```python
pip install Ensemble-GNN
```
## Usage

### Synthetic data

```python
from GNNSubNet import GNNSubNet as gnn

# Synthetic data set  ------------------------- #
loc   = "/home/bastian/GitHub/GNN-SubNet/GNNSubNet/datasets/synthetic"
ppi   = f'{loc}/NETWORK_synthetic.txt'
feats = [f'{loc}/FEATURES_synthetic.txt']
targ  = f'{loc}/TARGET_synthetic.txt'

# Read in the synthetic data and build a gnnsubnet object
g = gnn.GNNSubNet(loc, ppi, feats, targ, normalize=False)

# Get some general information about the data dimension
g.summary()

# import ensemble_gnn
import ensemble_gnn as egnn

# initialization: infer subnetworks and build ensemble
e1 = egnn.ensemble(g, niter=1) # niter=10 is recommended

# length of ensemble
len(e1.ensemble)

# train an gnn model on each subnetwork of the ensemble
e1.train()

# accuracy for each module
e1.train_accuracy

# the nodes of each ensemble member can be accessed via
e1.modules_gene_names 

# The first subnetwork used within the ensemble can be obtained from
e1.ensemble[0].dataset[0].edge_index

# grow the ensemble 
e1.grow(10)

# check the accuracy
e1.train_accuracy

# train again with a different train-validation split
e1.train()

# check the accuracy
e1.train_accuracy

# create a test set
g_test = gnn.GNNSubNet(loc, ppi, feats, targ, normalize=False)

# predict
e1.predict(g_test)

# predictions and accuracy for each subnetwork
# accuracy o the test set
e1.accuracy_test
# predictions on the test set
e1.predictions_test 
# true class labels of the test set
e1.true_class_test 

# the overall predictions based on the whole ensemble using majority vote
e1.predictions_test_mv

# lets check the performance 
from sklearn.metrics import accuracy_score
accuracy_score(e1.true_class_test[0], e1.predictions_test_mv)
```

### PPI network data

```python
from GNNSubNet import GNNSubNet as gnn

# location of the files
loc   = "/home/bastian/GitHub/GNN-SubNet/TCGA"
# PPI network
ppi   = f'{loc}/KIDNEY_RANDOM_PPI.txt'
# single-omic features
#feats = [f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# multi-omic features
feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# outcome class
targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'

# Load the multi-omics data 
g = gnn.GNNSubNet(loc, ppi, feats, targ)

# Get some general information about the data dimension
g.summary()

# import ensemble_gnn
import ensemble_gnn as egnn

# initialization: infer subnetworks and build ensemble
e1 = egnn.ensemble(g, niter=1) # niter=10 is recommended

# length of ensemble
len(e1.ensemble)

# train an gnn model on each subnetwork of the ensemble
e1.train()

# accuracy for each module
e1.train_accuracy

# the nodes of each ensemble member can be accessed via
e1.modules_gene_names 

# The first subnetwork used within the ensemble can be obtained from
e1.ensemble[0].dataset[0].edge_index

# grow the ensemble 
e1.grow(10)

# check the accuracy
e1.train_accuracy

# train again with a different train-validation split
e1.train()

# check the accuracy
e1.train_accuracy

# create a test set
# Load the multi-omics data 
g_test = gnn.GNNSubNet(loc, ppi, feats, targ)

# predict
e1.predict(g_test)

# predictions and accuracy for each subnetwork
# accuracy o the test set
e1.accuracy_test
# predictions on the test set
e1.predictions_test 
# true class labels of the test set
e1.true_class_test 

# the overall predictions based on the whole ensemble using majority vote
e1.predictions_test_mv

# lets check the performance 
from sklearn.metrics import accuracy_score
accuracy_score(e1.true_class_test[0], e1.predictions_test_mv)
```

## Citation
https://academic.oup.com/bioinformatics/article/38/Supplement_2/ii120/6702000

### Bibtex
```
@article{pfeifer2022gnn,
  title={Gnn-subnet: Disease subnetwork detection with explainable graph neural networks},
  author={Pfeifer, Bastian and Saranti, Anna and Holzinger, Andreas},
  journal={Bioinformatics},
  volume={38},
  number={Supplement\_2},
  pages={ii120--ii126},
  year={2022},
  publisher={Oxford University Press}
}

```
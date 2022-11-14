# Ensemble-GNN: ensemble learning with graph neural networks for disease subnetwork discovery

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

# train each model of the ensemble
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


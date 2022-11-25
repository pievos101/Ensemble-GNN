<p align="center">
<img src="https://github.com/pievos101/Ensemble-GNN/blob/main/logo2.png" width="400">
</p>

# Ensemble-GNN: ensemble learning with graph neural networks for disease module discovery

## Installation

The GNN-Subnet package is required 

https://github.com/pievos101/GNN-SubNet

To install GNNSubNet run:

```python
pip install torch==1.11.0 
pip install torchvision==0.12.0
pip install torch-geometric==2.0.4
pip install torch-scatter==2.0.9
pip install torch-sparse==0.6.13

# to install GNNSubNet from source (cloned GitHub repo) run:
pip install GNN-SubNet/
```
Preferred versions are: torch==1.11.0, torchvision==0.12.0, torch-geometric==2.0.4, torch-scatter==2.0.9, and torch-sparse==0.6.13.

To install Ensemble-GNN from source code (cloned GitHub repo) run:

```python
pip install Ensemble-GNN/
```
## Usage

### Synthetic data - Barabasi Networks

```python
# import GNNSubNet
from GNNSubNet import GNNSubNet as gnn
# import ensemble_gnn
import ensemble_gnn as egnn

# Synthetic data set  ------------------------- #
loc   = "/home/bastian/GitHub/GNN-SubNet/GNNSubNet/datasets/synthetic"
ppi   = f'{loc}/NETWORK_synthetic.txt'
feats = [f'{loc}/FEATURES_synthetic.txt']
targ  = f'{loc}/TARGET_synthetic.txt'

# Read in the synthetic data and build a gnnsubnet object
g = gnn.GNNSubNet(loc, ppi, feats, targ, normalize=False)

# Get some general information about the data dimension
g.summary()

# train-test split: 80-20 
g_train, g_test = egnn.train_test_split(g)

# initialization: infer subnetworks and build ensemble
e1 = egnn.ensemble(g_train, niter=1) # niter=10 is recommended

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

# predict via Majority Vote
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

# lets check the performance of the ensemble classifier
from sklearn.metrics import accuracy_score
accuracy_score(e1.true_class_test[0], e1.predictions_test_mv)
```

### Protein-Protein Interaction (PPI) network data

```python
# import GNNSubNet
from GNNSubNet import GNNSubNet as gnn
# import ensemble_gnn
import ensemble_gnn as egnn

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

# train-test split: 80-20 
g_train, g_test = egnn.train_test_split(g)

# initialization: infer subnetworks and build ensemble
e1 = egnn.ensemble(g_train, niter=1) # niter=10 is recommended

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
e1.grow(20)

# check the accuracy
e1.train_accuracy

# train again with a different train-validation split
e1.train()

# check the accuracy
e1.train_accuracy

# predict via Majority Vote
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

# lets check the performance of the ensemble classifier
from sklearn.metrics import accuracy_score
accuracy_score(e1.true_class_test[0], e1.predictions_test_mv)
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(e1.true_class_test[0], e1.predictions_test_mv)

# The results can be compared with non-ensemble-based classification

# train the GNN classifier 
g_train.train()

# predict
g_train.predict(g_test)

# lets check the performance of the non-ensemble classifier
from sklearn.metrics import accuracy_score
accuracy_score(e1.true_class_test[0], g_train.predictions_test)
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(e1.true_class_test[0], g_train.predictions_test)

```

# Federated Ensemble Learning with Ensemble-GNN

## Method 1

The models are collected from the clients and predictions are aggregated via Majority Vote. 


```python
# import GNNSubNet
from GNNSubNet import GNNSubNet as gnn
# import ensemble_gnn
import ensemble_gnn as egnn

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

# train-test split: 80-20 
g_train, g_test = egnn.train_test_split(g)

# 50 - 50 split of the training data
partie1, partie2 = egnn.split(g_train)

# train local ensemble classier of client 1
p1 = egnn.ensemble(partie1, niter=1)
p1.train()
#p1.grow(10)

# train local ensemble classier of client 2
p2 = egnn.ensemble(partie2, niter=1)
p2.train()
#p2.grow(10)

# aggregate the models from each client 
# without sharing any data
global_model = egnn.aggregate([p1,p2])

# Make predictions using the global model via Majority Vote
global_model.predict(g_test)

# lets check the performance of the federated ensemble classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

accuracy_score(global_model.true_class_test[0], global_model.predictions_test_mv)
balanced_accuracy_score(global_model.true_class_test[0], global_model.predictions_test_mv)

# Lets check the client-specific performances
# client 1 
p1.predict(g_test)
accuracy_score(p1.true_class_test[0], p1.predictions_test_mv)
balanced_accuracy_score(p1.true_class_test[0], p1.predictions_test_mv)

# client 2 
p2.predict(g_test)
accuracy_score(p2.true_class_test[0], p2.predictions_test_mv)
balanced_accuracy_score(p2.true_class_test[0], p2.predictions_test_mv)

```

## Method 2 

No models and data are shared, only the topologies of the relevant subnetworks. Coming soon ...


<p align="center">
<img src="https://github.com/pievos101/Ensemble-GNN/blob/main/fed_logo.png" width="400">
</p>

```python
# import GNNSubNet
from GNNSubNet import GNNSubNet as gnn
# import ensemble_gnn
import ensemble_gnn as egnn

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

partie1, partie2 = egnn.split(g)

# train local ensemble classier
p1 = egnn.ensemble(partie1, niter=1)
p1.train()
#p1.grow(10)

p2 = egnn.ensemble(partie2, niter=1)
p2.train()
#p2.grow(10)

# Partie1 suggests a subnet
subnet = p1.propose()

# Partie2 now checks whether this subnet is useful
# if yes, it will locally be included 

# p2.check(subnet)

```

# Miscellaneous

Logo made by Adobe Express Logo Maker: <https://www.adobe.com/express/create/logo>

## Citation
https://academic.oup.com/bioinformatics/article/38/Supplement_2/ii120/6702000

### Bibtex
```
@article{pfeifer2022gnn,
  title={{GNN-SubNet}: Disease subnetwork detection with explainable graph neural networks},
  author={Pfeifer, Bastian and Saranti, Anna and Holzinger, Andreas},
  journal={Bioinformatics},
  volume={38},
  number={Supplement\_2},
  pages={ii120--ii126},
  year={2022},
  publisher={Oxford University Press}
}

```

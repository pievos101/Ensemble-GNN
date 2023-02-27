<p align="center">
<img src="https://github.com/pievos101/Ensemble-GNN/blob/main/logo2.png" width="400">
</p>

# Ensemble learning with graph neural networks for disease module discovery

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

### Example 1: Synthetic data - Barabasi Networks

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
g_train, g_test = egnn.split(g, 0.8)

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

# grow the ensemble (greedy step)
# e1.grow(10)

# check the accuracy
e1.train_accuracy

# train again with a different train-validation split
# e1.train()

# check the accuracy
e1.train_accuracy

# predict via Majority Vote
predicted_class = e1.predict(g_test)

# the overall predictions based on the whole ensemble using majority vote
predicted_class
#e1.predictions_test_mv

# lets check the performance of the ensemble classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


acc = accuracy_score(g_test.true_class, predicted_class)
confusion_matrix(g_test.true_class,predicted_class)

from sklearn.metrics.cluster import normalized_mutual_info_score
normalized_mutual_info_score(g_test.true_class,predicted_class)

print("\n-----------")
print("Accuracy of ensemble classifier:", acc)

```

### Example 2: Protein-Protein Interaction (PPI) network data

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
#feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt']
# multi-omic features
feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# outcome class
targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'

# Load the multi-omics data
g = gnn.GNNSubNet(loc, ppi, feats, targ)

# Get some general information about the data dimension
g.summary()

# train-test split: 80-20
g_train, g_test = egnn.split(g, 0.8)

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

# grow the ensemble (greedy step)
# e1.grow(10) [optional]

# check the accuracy
e1.train_accuracy

# train again with a different train-validation split
# e1.train()

# check the accuracy
e1.train_accuracy

# predict via Majority Vote
predicted_class = e1.predict(g_test)

# the overall predictions based on the whole ensemble using majority vote
predicted_class

# lets check the performance of the ensemble classifier
print("\n-----------")
from sklearn.metrics import accuracy_score
acc = accuracy_score(g_test.true_class, predicted_class)
print("Accuracy of ensemble classifier:", acc)

from sklearn.metrics import balanced_accuracy_score
acc_bal = balanced_accuracy_score(g_test.true_class, predicted_class)
print("Balanced accuracy of ensemble classifier:", acc_bal)

from sklearn.metrics.cluster import normalized_mutual_info_score
nmi = normalized_mutual_info_score(g_test.true_class,predicted_class)
print("NMI of ensemble classifier:", nmi)


# The results can be compared with non-ensemble-based classification

# train the GNN classifier
g_train.train()

# predict
predicted_class = g_train.predict(g_test)

# lets check the performance of the non-ensemble classifier
from sklearn.metrics import accuracy_score
acc = accuracy_score(g_test.true_class, predicted_class)
print("Accuracy of ensemble classifier:", acc)

from sklearn.metrics import balanced_accuracy_score
acc_bal = balanced_accuracy_score(g_test.true_class, predicted_class)
print("Balanced accuracy of ensemble classifier:", acc_bal)

from sklearn.metrics.cluster import normalized_mutual_info_score
nmi = normalized_mutual_info_score(g_test.true_class,predicted_class)
print("NMI of ensemble classifier:", nmi)


```

# Federated Ensemble Learning with Ensemble-GNN

## Method 1

The models are collected from the clients and predictions are aggregated via Majority Vote.

<p align="center">
<img src="https://github.com/pievos101/Ensemble-GNN/blob/main/fed_logo1.png" width="500">
</p>

### Example 3
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
feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt']
# multi-omic features
#feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# outcome class
targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'

# Load the multi-omics data
g = gnn.GNNSubNet(loc, ppi, feats, targ)

# train-test split: 80-20
g_train, g_test = egnn.split(g, 0.8)

# 50 - 50 split of the training data
partie1, partie2 = egnn.split(g_train, 0.5)

# create local ensemble classier of client 1
p1 = egnn.ensemble(partie1, niter=1)
# train local ensemble classier of client 1
p1.train()
#p1.grow(10) # greedy step

# create local ensemble classier of client 2
p2 = egnn.ensemble(partie2, niter=1)
# train local ensemble classier of client 2
p2.train()
#p2.grow(10) # greedy step

# Lets check the client-specific performances
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

## client 1
p1_predicted_class = p1.predict(g_test)
acc1 = accuracy_score(g_test.true_class, p1_predicted_class)
acc1_bal = balanced_accuracy_score(g_test.true_class, p1_predicted_class)

## client 2
p2_predicted_class = p2.predict(g_test)
acc2 = accuracy_score(g_test.true_class, p2_predicted_class)
acc2_bal = balanced_accuracy_score(g_test.true_class, p2_predicted_class)

print("\n-----------")
print("Balanced accuracy of client 1 ensemble classifier:", acc1_bal)
print("Accuracy of client 1 ensemble classifier:", acc1)

print("\n-----------")
print("Balanced accuracy of client 2 ensemble classifier:", acc2_bal)
print("Accuracy of client 2 ensemble classifier:", acc2)

from sklearn.metrics.cluster import normalized_mutual_info_score
nmi1 = normalized_mutual_info_score(g_test.true_class,p1_predicted_class)
nmi2 = normalized_mutual_info_score(g_test.true_class,p2_predicted_class)

print("\n-----------")
print("NMI of client 1 ensemble classifier:", nmi1)

print("\n-----------")
print("NMI of client 2 ensemble classifier:", nmi2)


# aggregate the models from each client
# without sharing any data
global_model = egnn.aggregate([p1,p2])

# Make predictions using the global model via Majority Vote
predicted_class = global_model.predict(g_test)

# Lets check the performance of the federated ensemble classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

acc = accuracy_score(g_test.true_class, predicted_class)
acc_bal = balanced_accuracy_score(g_test.true_class, predicted_class)
print("\n-----------")
print("Balanced accuracy of global ensemble classifier:", acc_bal)
print("Accuracy of global ensemble classifier:", acc)

from sklearn.metrics.cluster import normalized_mutual_info_score
nmi = normalized_mutual_info_score(g_test.true_class,predicted_class)

print("\n-----------")
print("NMI of global ensemble classifier:", nmi)

```

### Example 4: Example code for multiple parties
```python3
#!/usr/bin/env python3
from GNNSubNet import GNNSubNet as gnn
import ensemble_gnn as egnn
import copy

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

# Number of parties
parties: int = 4

# train-test split: 80:20
g_train, g_test = egnn.split(g, 0.8)

# Split data equaliy with split_n and train single models
learned_ensembles: list = []
for party in egnn.split_n(g, parties):
    pn = egnn.ensemble(party, niter=1)
    pn.train()
    #pn.grow(10)
    learned_ensembles.append(pn)

# Aggregate all trained ensemble models
global_model = egnn.aggregate(learned_ensembles)

# lets check the performance of the federated ensemble classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
scores = [("Acc", accuracy_score), ("BalAcc", balanced_accuracy_score), ("MCC", matthews_corrcoef)]

predicted_class = global_model.predict(g_test)
print("# Global model performance")
print(", ".join(["%s: %.3f" % (c[0], c[1](g_test.true_class, predicted_class)) for c in scores]))

for party in range(0, len(learned_ensembles)):
    test = copy.deepcopy(g_test)
    pn_predicted_class = learned_ensembles[party].predict(test)
    print("# Party %d model performance" % (party+1))
    print(", ".join(["%s: %.3f" % (c[0], c[1](test.true_class, pn_predicted_class)) for c in scores]))
```

## Method 2 (in progress ...)

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
feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt']
# multi-omic features
#feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
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

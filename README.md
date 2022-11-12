# Ensemble-GNN: ensemble learning with graph neural networks

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

# Read in the synthetic data
g = gnn.GNNSubNet(loc, ppi, feats, targ, normalize=False)

# Get some general information about the data dimension
g.summary()

import ensemble_gnn as egnn

# init
e1 = egnn.ensemble(1)

e1.add(g)
e1.add(g)

e1.ensemble

e1.train()
e1.train_accuracy

```


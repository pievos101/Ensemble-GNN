#!/usr/bin/env python3
# import GNNSubNet
from GNNSubNet import GNNSubNet as gnn
# import ensemble_gnn
import ensemble_gnn as egnn

# Synthetic data set  ------------------------- #
loc   = "/home/roman/Schreibtisch/ENSEMBLE/GNN-SubNet/GNNSubNet/datasets/synthetic"
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
e1.train()

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

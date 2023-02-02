#!/usr/bin/env python3
# import GNNSubNet
from GNNSubNet import GNNSubNet as gnn
# import ensemble_gnn
import ensemble_gnn as egnn

# location of the files
loc   = "/home/roman/Schreibtisch/ENSEMBLE/GNN-SubNet/TCGA"
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
e1.grow(20)

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

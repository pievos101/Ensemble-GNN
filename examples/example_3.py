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

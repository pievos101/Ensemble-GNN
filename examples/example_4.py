#!/usr/bin/env python3
from GNNSubNet import GNNSubNet as gnn
import ensemble_gnn as egnn
import copy

# location of the files
loc   = "/home/roman/Schreibtisch/ENSEMBLE/GNN-SubNet/TCGA"
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

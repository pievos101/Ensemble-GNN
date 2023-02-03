#!/usr/bin/env python3

"""This script is almost an original file "example_4_b.py" created by Roman Martin."""


from GNNSubNet import GNNSubNet as gnn
import ensemble_gnn as egnn
import copy

import random
import time
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

RANDOM_SEED: int = 800

#
# location of the files
loc   = "/sybig/home/hch/FairPact/python-code/GNN-SubNet/TCGA/"
# PPI network
ppi   = f'{loc}/KIDNEY_RANDOM_PPI.txt'
# single-omic features
feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt']
# multi-omic features
# feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# outcome class
targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'

# location of the files
# loc   = "/sybig/home/hch/FairPact/python-code/Ensemble-GNN/datasets/TCGA-BRCA/"
# # PPI network
# ppi   = f'{loc}/HRPD_brca_subtypes.csv'
# # single-omic features
# #feats = [f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# # multi-omic features
# feats = [f'{loc}/GE_brca_subtypes.csv']
# # outcome class
# targ  = f'{loc}/target_brca_subtypes.csv'


# Number of parties
parties: int = 3
rounds: int = 4

# Split data equaliy with split_n and train single models
avg_local_performance: list = []
avg_ensemble_performance: list = []

# For reproducibility of the data splits
random.seed(RANDOM_SEED)
random_seeds: list = random.sample(range(100, 999), 2*rounds)

start = time.time()

# Repeat everything multiple times
for i in range(0, rounds):
    counter: int = 0
    learned_ensembles: list = []
    parties_testdata: list = []
    accuracy_single: list = []
    accuracy_ensemble: list = []

    print("# Round %d" % (i+1) )

    # Load the multi-omics data
    g = gnn.GNNSubNet(loc, ppi, feats, targ)
    print("## Total dataset length %d" % len(g.dataset))

    # Now each client learns his own ensemble
    for party in egnn.split_n(g, parties, random_seed=random_seeds[2*counter]): # 0, 2, 4
        counter += 1
        print("## Training party %d" % counter)
        g_train, g_test = egnn.split(party, 0.8, random_seed=random_seeds[2*counter - 1])  # 1, 3, 5
        print("### local train: %d, local test: %d" % (len(g_train.dataset), len(g_test.dataset)))
        pn = egnn.ensemble(g_train, niter=1)
        pn.train()
        predicted_local_classes = pn.predict(g_test)

        # Stores the test data and single client models into lists
        parties_testdata.append(g_test)
        learned_ensembles.append(pn)
        accuracy_single.append(balanced_accuracy_score(g_test.true_class, predicted_local_classes))

    print("## All balanced accuracy values from local tests: %s" % str(accuracy_single))

    # We are merging all ensembles together
    global_model = egnn.aggregate(learned_ensembles)

    # Each client applies the ensembled model on his own test data
    for party in range(0, len(learned_ensembles)):
        predicted_ensemble_classes = global_model.predict(parties_testdata[party])
        accuracy_ensemble.append(balanced_accuracy_score(parties_testdata[party].true_class, predicted_ensemble_classes))

    print("## All balanced accuracy values from global tests: %s" % str(accuracy_ensemble))
    avg_local: float = sum(accuracy_single)/len(accuracy_single)
    avg_ensembl: float = sum(accuracy_ensemble)/len(accuracy_ensemble)
    print("## Average performance with local model: %.3f and global model: %.3f" % (avg_local, avg_ensembl))
    avg_local_performance.append(avg_local)
    avg_ensemble_performance.append(avg_ensembl)

print("# Final result")
print("# Average performance over %d rounds with local model: %.3f and global model: %.3f" % (rounds, sum(avg_local_performance)/len(avg_local_performance), sum(avg_ensemble_performance)/len(avg_ensemble_performance)))

end = time.time()
print("\n\tTime to go through:", end-start)
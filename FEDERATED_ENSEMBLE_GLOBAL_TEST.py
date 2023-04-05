#!/usr/bin/env python3

from GNNSubNet import GNNSubNet as gnn
import ensemble_gnn as egnn
import copy

import random
import time
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

RANDOM_SEED: int = 232100


# location of the files
#loc   = "/home/bastian/GitHub/GNN-SubNet/TCGA"
# PPI network
#ppi   = f'{loc}/KIDNEY_RANDOM_PPI.txt'
# single-omic features
#feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt']
# multi-omic features
#feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# outcome class
#targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'


# # location of the files
# loc   = "/home/bastian/TCGA-BRCA"
# # PPI network
# ppi   = f'{loc}/HRPD_brca_subtypes.csv'
# # single-omic features
# feats = [f'{loc}/GE_brca_subtypes.csv']
# # outcome class
# targ  = f'{loc}/binary_target_brca_subtypes.csv'
# location of the files
loc   = "/sybig/home/hch/FairPact/python-code/Ensemble-GNN/datasets/TCGA-BRCA/"
# PPI network
ppi   = f'{loc}/HRPD_brca_subtypes.csv'
# single-omic features
#feats = [f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# multi-omic features
feats = [f'{loc}/GE_brca_subtypes.csv']
# outcome class
targ  = f'{loc}/binary_target_brca_subtypes.csv'



# Number of parties
parties: int = 3
rounds: int = 5

# Split data equaliy with split_n and train single models
avg_local_performance: list = []
avg_ensemble_performance: list = []

# For reproducibility of the data splits
random.seed(RANDOM_SEED)
random_seeds_parties: list = random.sample(range(100, 999), rounds) # rounds is on purpose here
random_seeds_rounds: list = random.sample(range(100, 999), rounds)

start = time.time()

# Repeat everything multiple times
for i in range(0, rounds):
    counter: int = 0
    learned_ensembles: list = []
    parties_testdata: list = []
    accuracy_single: list = []
    accuracy_ensemble: list = []

    print("# Round %d" % (i+1))

    # Load the multi-omics data
    g = gnn.GNNSubNet(loc, ppi, feats, targ, normalize=False)
    print("## Total dataset length %d" % len(g.dataset))

    # Global test set
    g_train, g_test = egnn.split(g, 0.8, random_seed=random_seeds_rounds[i])

    # Now each client learns his own ensemble
    participants = egnn.split_n(g_train, parties, random_seed=random_seeds_parties[i])

    for party in participants: # 0, 2, 4
        counter += 1
        print("## Training party %d" % counter)
        # print("### local train: %d, local test: %d" % (len(g_train.dataset), len(g_test.dataset)))
        pn = egnn.ensemble(party, niter=1, method="graphcheb", epoch_nr=60, verbose=1)
        pn.train(epoch_nr=60)
        predicted_local_classes = pn.predict(g_test)

        # Stores the test data and single client models into lists
        parties_testdata.append(g_test)
        learned_ensembles.append(pn)
        accuracy_single.append(balanced_accuracy_score(g_test.true_class, predicted_local_classes))

    print("## All balanced accuracy values from local tests: %s" % str(accuracy_single))

    # We are merging all ensembles together
    global_model = egnn.aggregate(learned_ensembles)
    predicted_ensemble_classes = global_model.predict(g_test)
    accuracy_ensemble.append(balanced_accuracy_score(g_test.true_class, predicted_ensemble_classes))

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
print("Average local perfromance:", avg_local_performance)
print("Average ensemble perfromance:", avg_ensemble_performance)


#In [12]: avg_local_performance
#Out[12]: 
#[0.7760112593828191,
# 0.7954371698637753,
# 0.7740825688073395,
# 0.7656727828746176,
# 0.7622150403113706,
# 0.7805984153461218,
# 0.7897032249096471,
# 0.7921705587989992,
# 0.7955066722268557,
# 0.7955935501807061]

#In [13]: avg_ensemble_performance
#Out[13]: 
#[0.8154190992493745,
# 0.8309007506255213,
# 0.8193286071726439,
# 0.8005629691409508,
# 0.8593098415346121,
# 0.8245934111759801,
# 0.8224040867389492,
# 0.8202147623019183,
# 0.8365825688073394,
# 0.8121351125938282]

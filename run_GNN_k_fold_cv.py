#!/usr/bin/env python3

"""This script is an adaptation of the original file "example_4_b.py" created by Roman Martin."""


from GNNSubNet import GNNSubNet as gnn
import ensemble_gnn as egnn
import copy

import random
import time
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

RANDOM_SEED: int = 800


# # location of the files
# loc   = "/sybig/home/hch/FairPact/python-code/GNN-SubNet/TCGA/"
# # PPI network
# ppi   = f'{loc}/KIDNEY_RANDOM_PPI.txt'
# # single-omic features
# feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt']
# # multi-omic features
# # feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# # outcome class
# targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'

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


# Number of splits for K-fold cross validation
splits: int = 10

# Split data equaliy with split_n and train single models
avg_local_performance: list = []
avg_ensemble_performance: list = []

# For reproducibility of the data splits
# random.seed(RANDOM_SEED)
# random_seeds: list = random.sample(range(100, 999), 2*rounds)

start = time.time()
# Load the multi-omics data
g = gnn.GNNSubNet(loc, ppi, feats, targ, verbose=2, normalize=False)

# Get some general information about the data dimension
# g.summary()

accuracy_single: list = []
counter: int = 0

model_pairs: list = egnn.split_n_fold_cv(g, n_splits=splits, random_seed=RANDOM_SEED)

for g_train, g_test in model_pairs:
    counter += 1
    print("## Training fold %d" % counter)
    g_train.train()
    predicted_local_classes = g_train.predict(g_test)
    print("### Balanced accuracy: fold %d score: %.3f" % (counter, balanced_accuracy_score(g_test.true_class, predicted_local_classes)))
    # print("## Finished training fold %d" % counter)
    # Stores the test data and single client models into lists
    accuracy_single.append(balanced_accuracy_score(g_test.true_class, predicted_local_classes))

avg_local: float = sum(accuracy_single)/len(accuracy_single)

print("# All balanced accuracy values from local tests: %s" % str(accuracy_single))
print("# Average performance with local model: %.3f" % (avg_local))

end = time.time()
print("\n\tTime to go through:", end-start)
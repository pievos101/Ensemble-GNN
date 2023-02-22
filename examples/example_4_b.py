#!/usr/bin/env python3
from GNNSubNet import GNNSubNet as gnn

import ensemble_gnn as egnn
import copy
from random import randint

from sklearn.metrics import average_precision_score

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

# Number of parties
parties: int = 3
rounds: int = 3
benchmarker = egnn.Benchmarker(parties, rounds, federated=True, scorer=[("APS", average_precision_score)])

# Repeat everything multiple times
for round in range(0, rounds):
    counter: int = 0
    learned_ensembles: list = []
    parties_testdata: list = []

    print("# Round %d" % (round+1) )

    # Load the multi-omics data
    g = gnn.GNNSubNet(loc, ppi, feats, targ)
    print("## Total dataset length %d" % len(g.dataset))

    # Now each client learns his own ensemble
    for party in egnn.split_n( g, parties, random_seed=randint(100,999) ):
        counter += 1
        print("## Training party %d" % counter)
        g_train, g_test = egnn.split(party, 0.8, random_seed=randint(100,999))
        print("### local train: %d, local test: %d" % (len(g_train.dataset), len(g_test.dataset)))
        pn = egnn.ensemble(g_train, niter=1)
        pn.train()
        predicted_local_classes = pn.predict(g_test)

        # Stores the test data and single client models into lists
        parties_testdata.append(g_test)
        learned_ensembles.append(pn)
        benchmarker.set_scores(g_test.true_class, predicted_local_classes, round)

    #print("## All single scores from local tests: %s" % str(benchmarker.get_single_results(round)))

    # We are merging all ensembles together
    global_model = egnn.aggregate(learned_ensembles)

    # Each client applies the ensembled model on his own test data
    for party in range(0, len(learned_ensembles)):
        predicted_ensemble_classes = global_model.predict(parties_testdata[party])
        benchmarker.set_scores(parties_testdata[party].true_class, predicted_ensemble_classes, round, federated=True)

    #print("## All single scores from global tests: %s" % str(benchmarker.get_single_results(round, federated=True)))
    benchmarker.calculate_round_average(round)

benchmarker.report()

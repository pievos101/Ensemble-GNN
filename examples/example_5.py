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

partie1, partie2 = egnn.split(g, 0.8)

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

# utils 
from torch_geometric.data.data import Data
import torch
from copy import copy
import numpy as np
import random

def train_test_split(gnnsubnet):
	""""
	Compute train test 
	"""
	gnn1_train = copy(gnnsubnet)
	gnn2_test  = copy(gnnsubnet)

	dataset_list = copy(gnnsubnet.dataset)
	random.shuffle(dataset_list)
	list_len = len(dataset_list)
	train_set_len = int(list_len * 4 / 5)

	train_dataset_list = dataset_list[:train_set_len]
	test_dataset_list  = dataset_list[train_set_len:]
	gnn1_train.dataset = train_dataset_list
	gnn2_test.dataset  = test_dataset_list

	return gnn1_train, gnn2_test
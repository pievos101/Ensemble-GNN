# utils 
from torch_geometric.data.data import Data
import torch
import copy
import numpy as np
import random


def train_test_split(gnnsubnet):
	""""
	Compute train test 
	"""
	gnn1_train = copy.deepcopy(gnnsubnet) # deep copy?
	gnn2_test  = copy.deepcopy(gnnsubnet) # deep copy?

	dataset_list = copy.deepcopy(gnnsubnet.dataset)
	random.shuffle(dataset_list)
	list_len = len(dataset_list)
	train_set_len = int(list_len * 4 / 5)

	train_dataset_list = dataset_list[:train_set_len]
	test_dataset_list  = dataset_list[train_set_len:]
	gnn1_train.dataset = train_dataset_list
	gnn2_test.dataset  = test_dataset_list

	return gnn1_train, gnn2_test

def split(gnnsubnet):
	""""
	50-50 split 
	"""
	gnn1_a  = copy.deepcopy(gnnsubnet) # deep copy ?
	gnn2_b  = copy.deepcopy(gnnsubnet) # deep copy ?

	dataset_list = copy.deepcopy(gnnsubnet.dataset)
	random.shuffle(dataset_list)
	list_len = len(dataset_list)
	set_len = int(list_len * 1 / 2)

	a_dataset_list = dataset_list[:set_len]
	b_dataset_list = dataset_list[set_len:]
	gnn1_a.dataset = a_dataset_list
	gnn2_b.dataset = b_dataset_list

	return gnn1_a, gnn2_b

def aggregate(models_list):
	# import GNNSubNet
	from GNNSubNet import GNNSubNet as gnn
	# import ensemble_gnn
	import ensemble_gnn as egnn

	e_all = egnn.ensemble()
	for xx in range(len(models_list)):
		e = models_list[xx].send_model()
		for yy in range(len(e)):
			e_all.ensemble.append(e[yy])
	return e_all

	
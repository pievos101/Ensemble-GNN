# utils
from torch_geometric.data.data import Data
from GNNSubNet import GNNSubNet
from GNNSubNet import GNNSubNet as gnn
import ensemble_gnn as egnn
import copy
import numpy as np
import random


def split(gnnsubnet: GNNSubNet, split: float = 0.8, random_seed: int = 42) -> tuple:
	""""
	Split dataset from GNNSubNet into multiple two datasets

	:param GNNSubNet gnnsubnet: 	A GNNSubNet contains the dataset
	:param float	 split:		 	Splitting ratio
	:param int		 random_seed:	Seed for random shuffling
	:return tuple:					Two GNNSubNets with different datasets
	"""
	gnn1_train: GNNSubNet = copy.deepcopy(gnnsubnet)
	gnn2_test:  GNNSubNet = copy.deepcopy(gnnsubnet)

	dataset_list = copy.deepcopy(gnnsubnet.dataset)
	random.seed(random_seed)
	random.shuffle(dataset_list)
	list_len: int = len(dataset_list)
	train_set_len: int = int(list_len * split)

	train_dataset_list: list = dataset_list[:train_set_len]
	test_dataset_list: list  = dataset_list[train_set_len:]
	gnn1_train.dataset = train_dataset_list
	label = []
	for xx in range(len(train_dataset_list)): label.append(train_dataset_list[xx].y)
	gnn1_train.true_class = np.array(label)

	gnn2_test.dataset  = test_dataset_list
	label = []
	for xx in range(len(test_dataset_list)): label.append(test_dataset_list[xx].y)
	gnn2_test.true_class = np.array(label)

	return gnn1_train, gnn2_test

def split_n(gnnsubnet: GNNSubNet, parties: int = 2, proportions: list = None, random_seed: int = 42) -> list:
	"""
	Split dataset from GNNSubnet into multiple fractions

	:param GNNSubNet gnnsubnet:  	The given given GNNSubNet contains the dataset
	:param int		 parties:	 	The number of resulted splits
	:param list		 proportions:	A list of floats with the split proportions
								 	if not equally splitted. sum(proportions) = 1
	:param int		random_seed:	Seed for random shuffling
	:return list:					A list of splits
	"""

	gnn_subnets: list = []
	dataset_list = copy.deepcopy(gnnsubnet.dataset)
	print(gnn_data_list)
	random.seed(random_seed)
	random.shuffle(dataset_list)

	if proportions is not None and type(proportions) == list and \
		len(proportions) > 1 and sum(proportions) == 1:
		pass

	for p in parties:
		pass

	return []


def aggregate(models_list: list):
	"""
	Aggregate multiple graphs to an ensemble

	:param list	models_list:	A list of GNNSubNet objects
	"""
	e_all = egnn.ensemble()
	for xx in range(len(models_list)):
		e = models_list[xx].send_model()
		for yy in range(len(e)):
			e_all.ensemble.append(e[yy])
	return e_all

# utils
from torch_geometric.data.data import Data
from sklearn.model_selection import KFold
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
	gnn1_train.true_class = np.array([train_dataset_list[xx].y for xx in range(len(train_dataset_list))])
	gnn2_test.dataset  = test_dataset_list
	gnn2_test.true_class = np.array([test_dataset_list[xx].y for xx in range(len(test_dataset_list))])

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
	counter: int = 0

	dataset_list = copy.deepcopy(gnnsubnet.dataset)
	random.seed(random_seed)
	random.shuffle(dataset_list)

	# Reset if proportions does not fit the requirements
	if not(proportions is not None and type(proportions) == list and \
		len(proportions) > 1 and sum(proportions) == 1):
		proportions = [1/parties for fraction in range(0, parties)]

	ranges: list = [round(len(dataset_list)*r) for r in proportions]
	# change last element if the sum of rounded intervals is not equal the length of the dataset
	if sum(ranges) != len(dataset_list):
		ranges[len(ranges)-1] = len(dataset_list) - sum(ranges[:-1])

	for p in range(0, parties):
		# print("%d: %d:%d" % (p, counter, counter+ranges[p]))
		gnn: GNNSubNet = copy.deepcopy(gnnsubnet)
		gnn.dataset = dataset_list[counter:counter+ranges[p]]
		gnn.true_class = np.array([gnn.dataset[t].y for t in range(len(gnn.dataset))])
		counter += ranges[p]
		gnn_subnets.append(gnn)

	return gnn_subnets


def split_n_fold_cv(gnnsubnet: GNNSubNet, n_splits: int = 3, random_seed: int = 42) -> list:
	"""
	Split dataset from GNNSubnet into fractions for n_splits-fold crossvalidation
	:param GNNSubNet gnnsubnet:  	The given GNNSubNet object contains the dataset
	:param int		 n_splits:	 	The number of splits for k-fold cross validation
	:param int		random_seed:	Seed for random shuffling
	:return list:					A list of object pairs [gnn_train, gnn_test], objects are of type GNNSubNet
	"""

	gnn_subnets: list = []

	dataset_list = copy.deepcopy(gnnsubnet.dataset)
	random.seed(random_seed)
	random.shuffle(dataset_list)

	kf = KFold(n_splits=n_splits)
	for train, test in kf.split(dataset_list):
		gnn_train: GNNSubNet = copy.deepcopy(gnnsubnet)
		gnn_test: GNNSubNet = copy.deepcopy(gnnsubnet)
		gnn_train.dataset = [dataset_list[t] for t in train]
		gnn_test.dataset = [dataset_list[t] for t in test]
		gnn_train.true_class = np.array([gnn_train.dataset[t].y for t in range(len(gnn_train.dataset))])
		gnn_test.true_class = np.array([gnn_test.dataset[t].y for t in range(len(gnn_test.dataset))])
		gnn_subnets.append([gnn_train, gnn_test])

	return gnn_subnets


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

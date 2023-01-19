# utils
from torch_geometric.data.data import Data
from GNNSubNet import GNNSubNet
import torch
import copy
import numpy as np
import random


def train_test_split(gnnsubnet: GNNSubNet, split: float = 0.8) -> tuple:
	""""
	Split dataset from GNNSubNet into multiple training and test dataset

	:param GNNSubNet gnnsubnet: The given given GNNSubNet contains the dataset
	:param float	 split:		The provided training-to-test ratio
	:return tuple:				Two GNNSubNets with different datasets
	"""
	gnn1_train: GNNSubNet = copy.deepcopy(gnnsubnet)
	gnn2_test:  GNNSubNet = copy.deepcopy(gnnsubnet)

	dataset_list: GNNSubNet = copy.deepcopy(gnnsubnet.dataset)
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


	print("##SPLIT##")
	print(len(gnn1_a.dataset))
	print(len(gnn2_b.dataset))

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

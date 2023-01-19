# utils
from torch_geometric.data.data import Data
from GNNSubNet import GNNSubNet
import torch
import copy
import numpy as np
import random


def split(gnnsubnet: GNNSubNet, split: float = 0.8) -> tuple:
	""""
	Split dataset from GNNSubNet into multiple two datasets

	:param GNNSubNet gnnsubnet: The given given GNNSubNet contains the dataset
	:param float	 split:		The provided splitting ratio
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

	print("##TRAIN SPLIT##")
	print(gnn1_train.true_class)
	print(gnn2_test.true_class)
	gnn1_train.summary

	return gnn1_train, gnn2_test

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

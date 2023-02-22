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


class Benchmarker:
	from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, auc, matthews_corrcoef
	"""
	Manages average score values over multiple parties and multiple iterations(rounds)
	"""

	scores = [
		("BalAcc", balanced_accuracy_score),
		("MCC", matthews_corrcoef),
		("F1", f1_score)
	]
	"""
	Predefined scoring methods
	"""

	def __init__(self, party: int = 1, rounds: int = 3, federated: bool = False, verbose: int = 0, scorer: list = []):
		"""
		Initialize the Benchmarker by creating a bucket data structure to store all single score values
		:param 	int		party: 		Number of parties
		:param	int		rounds: 	Number of iterations/rounds
		:param	bool	federated: 	True if a global and local model are expected
		:param	int		verbose: 	Level of verbose logs. 0 = mostly silent
		:param	list	scorer:		Additional scoring methods. List of tuples with abbrevation and scoring function
		"""
		self.party: int  = party
		self.federated: bool = federated
		self.rounds: int = rounds
		self.verbose: int = verbose

		if type(scorer) == list and len(scorer):
			for sc in scorer:
				if len(sc) == 2 and type(sc[0]) == str and callable(sc[1]):
					self.scores.append(sc)

		self.bucket: dict = self.__create_bucket()

	def __create_bucket(self) -> dict:
		"""
		Preparing data structure to store all values. Scores * Rounds * Parties
		"""
		bucket: dict = {"avg_local": {}, "scores": { "local": {}}}
		[bucket["avg_local"].update({i[0]: []}) for i in self.scores]
		[bucket["scores"]["local"].update({i: {}}) for i in range(0, self.rounds)]

		if self.federated:
			bucket.update({"avg_federated": {}})
			bucket["scores"].update({"federated": {}})
			[bucket["avg_federated"].update({i[0]: []}) for i in self.scores]
			[bucket["scores"]["federated"].update({i: {}}) for i in range(0, self.rounds)]

		for round in range(0, self.rounds):
			[bucket["scores"]["local"][round].update({i[0]: []}) for i in self.scores]
			if self.federated:
				[bucket["scores"]["federated"][round].update({i[0]: []}) for i in self.scores]

		return bucket

	def calculate_round_average(self, round: int = 0):
		"""
		Compute each round average. Necessary to execute after finishing a interation/round
		:param	int	round:	The current round number
		"""
		for score in self.scores:
			self.bucket["avg_local"][score[0]].append(sum(self.bucket["scores"]["local"][round][score[0]])/len(self.bucket["scores"]["local"][round][score[0]]))
			if self.federated:
				self.bucket["avg_federated"][score[0]].append(sum(self.bucket["scores"]["federated"][round][score[0]])/len(self.bucket["scores"]["federated"][round][score[0]]))

	def set_scores(self, y_true, y_pred, round: int = 0, federated: bool = False):
		"""
		Stores the score of the given values
		:param			y_true:		the truth y values
		:param			y_pred:		the predicted y values
		:param	int		round:		The current round number
		:param	bool	federated:	True for global model
		"""
		bucket_class: str = "federated" if federated else "local"
		for score in self.scores:
			self.bucket["scores"][bucket_class][round][score[0]].append(score[1](y_true, y_pred))

	def get_single_results(self, round: int = 0, federated: bool = False) -> dict:
		"""
		Returns all score numbers of a specific round
		:param	int		round:		The current round number
		:param	bool	federated:	True for global model
		:return	dict				The dictionary with all values for a single round
		"""
		bucket_class: str = "federated" if federated else "local"
		return self.bucket["scores"][bucket_class][round]

	def get_avg_score(self, scorer: str, federated: bool = False) -> float:
		"""
		Returns the total averaged value for a specific scoring method
		:param	str		scorer:		The scoring name
		:param	bool	federated:	True for global model
		"""
		bucket_class: str = "avg_federated" if federated else "avg_local"
		return sum(self.bucket[bucket_class][scorer])/len(self.bucket[bucket_class][scorer])

	def get_avg_round(self, round: int = 0):
		"""
		Prints the average of all scoring methods from a round
		:param	int	round: The current round number
		"""
		print("# Round %d performance" % round+1)
		if self.federated and self.verbose:
			for score in self.scores:
				print("## %s: local model %.3f and global model %.3f" % (score[0], self.bucket["avg_local"][round], self.bucket["avg_federated"][round]))
		elif self.verbose:
			for score in self.scores:
				print("## %s: %.3f" % (score[0], self.bucket["avg_local"][round]))

	def report(self):
		"""
		Prints all averages
		"""
		print("# Total performance after %d rounds" % self.rounds)
		if self.federated:
			for score in self.scores:
				print("# %s: local model %.3f and global model %.3f" % (score[0], self.get_avg_score(score[0]), self.get_avg_score(score[0], federated=True)))
		else:
			for score in self.scores:
				print("# %s: %.3f" % (score[0], self.get_avg_score(score[0])))

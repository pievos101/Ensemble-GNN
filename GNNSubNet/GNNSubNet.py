# GNNSubNet.py
# Authors: Bastian Pfeifer <https://github.com/pievos101>, Marcus D. Bloice <https://github.com/mdbloice>
from urllib.parse import _NetlocResultMixinStr
import numpy as np
import random
#from scipy.sparse.extract import find
from scipy.sparse import find
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn.modules import conv
from torch_geometric import data
from torch_geometric.data import DataLoader, Batch
from pathlib import Path
import copy
from tqdm import tqdm
import os
import requests
import pandas as pd
import io
#from collections.abc import Mapping

from torch_geometric.data.data import Data

from .gnn_training_utils import check_if_graph_is_connected, pass_data_iteratively
from .dataset import generate, load_OMICS_dataset, convert_to_s2vgraph
from .gnn_explainer import GNNExplainer
from .graphcnn import GraphCNN

from .community_detection import find_communities
from .edge_importance import calc_edge_importance

class GNNSubNet(object):
    """
    The class GNNSubSet represents the main user API for the
    GNN-SubNet package.
    """
    def __init__(self, location=None, ppi=None, features=None, target=None, cutoff=950, normalize=True, verbose: int = 0, initialize: bool = True) -> None:

        self.location = location
        self.ppi = ppi
        self.features = features
        self.target = target
        self.dataset = None
        self.model_status = None
        self.model = None
        self.gene_names = None
        self.accuracy = None
        self.confusion_matrix = None
        self.test_loss = None

        self.verbose = verbose
        self.cutoff = cutoff
        self.normalize = normalize
        self.initialize = initialize

        self.true_class = None
        self.s2v_test_dataset = None
        self.edge_mask = None
        self.node_mask = None
        self.node_mask_matrix = None
        self.modules = None
        self.module_importances = None

        # Flags for internal use (hidden from user)
        self._explainer_run = False

        if ppi == None:
            return None

        if self.initialize:
            self.initialize_graph()


    def initialize_graph(self):
        dataset, gene_names = load_OMICS_dataset(self.ppi, self.features, self.target, True, self.cutoff, self.normalize)

         # Check whether graph is connected
        check = check_if_graph_is_connected(dataset[0].edge_index)
        print("Graph is connected ", check)

        if check == False:

            print("Calculate subgraph ...")
            dataset, gene_names = load_OMICS_dataset(self.ppi, self.features, self.target, False, self.cutoff, self.normalize)

        check = check_if_graph_is_connected(dataset[0].edge_index)
        print("Graph is connected ", check)

        if self.verbose:
            print("# DATASET LOADED")

        self.dataset = dataset
        self.gene_names = gene_names
        self.edges =  np.transpose(np.array(dataset[0].edge_index))


    def summary(self):
        """
        Print a summary for the GNNSubSet object's current state.
        """
        print("")
        print("Number of nodes:", len(self.dataset[0].x))
        print("Number of edges:", self.edges.shape[0])
        print("Number of modalities:",self.dataset[0].x.shape[1])

        #for i in self.__dict__:
        #    print('%s: %s' % (i, self.__dict__[i]))

    def create_mini_batches(s2v_train_dataset, batch_size):
        mini_batches = []
        data = s2v_train_dataset
        random.shuffle(data)

        n_minibatches = len(data) // batch_size
        i = 0

        # mini_batches = [data[i * batch_size:(i + 1) * batch_size]]
        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1) * batch_size]
            mini_batches.append(mini_batch)
        if len(data) % batch_size != 0:
            mini_batch = data[i * batch_size:len(data)]
            mini_batches.append(mini_batch)
        return mini_batches


    def train(self, epoch_nr = 24, shuffle=True, weights=False):
        """
        Train the GNN model on the data provided during initialisation.
        """
        use_weights = False

        dataset = self.dataset
        gene_names = self.gene_names

        graphs_class_0_list = []
        graphs_class_1_list = []
        for graph in dataset:
            if graph.y.numpy() == 0:
                graphs_class_0_list.append(graph)
            else:
                graphs_class_1_list.append(graph)

        graphs_class_0_len = len(graphs_class_0_list)
        graphs_class_1_len = len(graphs_class_1_list)

        if self.verbose >= 1:
            print(f"## Graphs class 0: {graphs_class_0_len}, Graphs class 1: {graphs_class_1_len}")

        ########################################################################################################################
        # Downsampling of the class that contains more elements ===========================================================
        # ########################################################################################################################

        if graphs_class_0_len >= graphs_class_1_len:
            random_graphs_class_0_list = random.sample(graphs_class_0_list, graphs_class_1_len)
            balanced_dataset_list = graphs_class_1_list + random_graphs_class_0_list

        if graphs_class_0_len < graphs_class_1_len:
            random_graphs_class_1_list = random.sample(graphs_class_1_list, graphs_class_0_len)
            balanced_dataset_list = graphs_class_0_list + random_graphs_class_1_list

        #print(len(random_graphs_class_0_list))
        #print(len(random_graphs_class_1_list))

        random.shuffle(balanced_dataset_list)
        if self.verbose >= 1:
            print(f"## Length of balanced dataset list: {len(balanced_dataset_list)}")

        list_len = len(balanced_dataset_list)
        #print(list_len)
        train_set_len = int(list_len * 4 / 5)
        train_dataset_list = balanced_dataset_list[:train_set_len]
        test_dataset_list  = balanced_dataset_list[train_set_len:]

        train_graph_class_0_nr = 0
        train_graph_class_1_nr = 0
        for graph in train_dataset_list:
            if graph.y.numpy() == 0:
                train_graph_class_0_nr += 1
            else:
                train_graph_class_1_nr += 1

        if self.verbose >= 1:
            print(f"## Train graph class 0: {train_graph_class_0_nr}, train graph class 1: {train_graph_class_1_nr}")

        test_graph_class_0_nr = 0
        test_graph_class_1_nr = 0
        for graph in test_dataset_list:
            if graph.y.numpy() == 0:
                test_graph_class_0_nr += 1
            else:
                test_graph_class_1_nr += 1
        if self.verbose >= 1:
            print(f"## Validation graph class 0: {test_graph_class_0_nr}, validation graph class 1: {test_graph_class_1_nr}")

        s2v_train_dataset = convert_to_s2vgraph(train_dataset_list)
        s2v_test_dataset  = convert_to_s2vgraph(test_dataset_list)

        import pickle
        # pickle.dump(s2v_train_dataset, open("./docs/train_brca.p", "wb"))
        # pickle.dump(s2v_test_dataset, open("./docs/test_brca.p", "wb"))

        # s2v_train_dataset = pickle.load(open("./docs/train.p", "rb"))
        # s2v_test_dataset = pickle.load(open("./docs/test.p", "rb"))

        # s2v_train_dataset = pickle.load(open("./docs/train_brca.p", "rb"))
        # s2v_test_dataset = pickle.load(open("./docs/test_brca.p", "rb"))


        # TRAIN GNN -------------------------------------------------- #
        #count = 0
        #for item in dataset:
        #    count += item.y.item()

        #weight = torch.tensor([count/len(dataset), 1-count/len(dataset)])
        #print(count/len(dataset), 1-count/len(dataset))

        model_path = 'omics_model.pth'
        no_of_features = dataset[0].x.shape[1]
        nodes_per_graph_nr = dataset[0].x.shape[0]

        #print(len(dataset), len(dataset)*0.2)
        #s2v_dataset = convert_to_s2vgraph(dataset)
        #train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=123)
        #s2v_train_dataset = convert_to_s2vgraph(train_dataset)
        #s2v_test_dataset = convert_to_s2vgraph(test_dataset)
        #s2v_train_dataset, s2v_test_dataset = train_test_split(s2v_dataset, test_size=0.2, random_state=123)

        input_dim = no_of_features
        n_classes = 2

        # TODO: put hyperparameters as external parameters
        model = GraphCNN(5, 2, input_dim, 32, n_classes, 0, True, 'sum1', 'sum', 0)
        opt = torch.optim.Adam(model.parameters(), lr=0.01) #, weight_decay=0.11) # 25 epochs for KIRC

        load_model = False
        if load_model:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            opt = checkpoint['optimizer']

        model.train()

        # TODO: put hyperparameters as external parameters
        min_loss = 50
        best_model = GraphCNN(5, 2, input_dim, 32, n_classes, 0, True, 'sum1', 'sum', 0)
        min_val_loss = 15000000 # need large number for initialization of min_val_loss with the first epoch
        n_epochs_stop = 50 # to stop training if no improvements in n_epochs_stop epochs
        epochs_no_improve = 0
        batch_size = 16

        # steps_per_epoch = 10
        # print("training process")
        # print(type(s2v_train_dataset))
        for epoch in range(epoch_nr):
            model.train()
            mini_batches = GNNSubNet.create_mini_batches(s2v_train_dataset, batch_size)

            # print("\tlen(mini_batches)", len(mini_batches))

            mini_batches_pbar = tqdm(mini_batches, unit='batch', disable=not self.verbose)
            epoch_loss = 0

            for batch_graph in mini_batches_pbar:
                # selected_idx = np.random.permutation(len(s2v_train_dataset))[:32]
                # batch_graph = [s2v_train_dataset[idx] for idx in selected_idx]
                logits = model(batch_graph)
                labels = torch.LongTensor([graph.label for graph in batch_graph])
                # if use_weights:
                #     loss = nn.CrossEntropyLoss(weight=weight)(logits,labels)
                # else:
                #     loss = nn.CrossEntropyLoss()(logits,labels)
                loss = nn.CrossEntropyLoss()(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.detach().item()

            epoch_loss /= len(mini_batches)
            model.eval()
            output = pass_data_iteratively(model, s2v_train_dataset)
            predicted_class = output.max(1, keepdim=True)[1]
            labels = torch.LongTensor([graph.label for graph in s2v_train_dataset])
            correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
            acc_train = correct / float(len(s2v_train_dataset))
            if self.verbose >= 1:
                print('## Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            if self.verbose:
                print(f"# Train Acc {acc_train:.4f}")

            mini_batches_pbar.set_description('epoch: %d' % (epoch))
            val_loss = 0
            output = pass_data_iteratively(model, s2v_test_dataset)

            pred = output.max(1, keepdim=True)[1]
            labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
            if use_weights:
                    loss = nn.CrossEntropyLoss(weight=weight)(output,labels)
            else:
                loss = nn.CrossEntropyLoss()(output,labels)
            val_loss += loss

            if self.verbose >= 1:
                print('Epoch {}, val_loss {:.4f}'.format(epoch, val_loss))
            if val_loss < min_val_loss:
                if self.verbose:
                    print(f"Saving best model with validation loss {val_loss}")
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
                min_val_loss = val_loss

            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    if self.verbose >= 1:
                        print('## Early stopping!')
                    model.load_state_dict(best_model.state_dict(), strict=False)
                    break

        confusion_array = []
        true_class_array = []
        predicted_class_array = []
        model.eval()
        correct = 0
        true_class_array = []
        predicted_class_array = []

        test_loss = 0

        model.load_state_dict(best_model.state_dict(), strict=False)

        output = pass_data_iteratively(model, s2v_test_dataset)
        predicted_class = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
        correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
        acc_test = correct / float(len(s2v_test_dataset))

        if use_weights:
            loss = nn.CrossEntropyLoss(weight=weight)(output,labels)
        else:
            loss = nn.CrossEntropyLoss()(output,labels)
        test_loss = loss

        predicted_class_array = np.append(predicted_class_array, predicted_class)
        true_class_array = np.append(true_class_array, labels)

        confusion_matrix_gnn = confusion_matrix(true_class_array, predicted_class_array)

        if self.verbose >= 1:
            print("\nConfusion matrix (Validation set):\n")
            print(confusion_matrix_gnn)


        counter = 0
        for it, i in zip(predicted_class_array, range(len(predicted_class_array))):
            if it == true_class_array[i]:
                counter += 1

        accuracy = counter/len(true_class_array) * 100

        if self.verbose >= 1:
            print("Validation accuracy: {}%".format(accuracy))
            print("Validation loss {}".format(test_loss))

        checkpoint = {
            'state_dict': best_model.state_dict(),
            'optimizer': opt.state_dict()
        }
        torch.save(checkpoint, model_path)

        model.train()

        self.model_status = 'Trained'
        self.model = copy.deepcopy(model)
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix_gnn
        self.test_loss = test_loss
        self.s2v_test_dataset = s2v_test_dataset
        self.predictions = predicted_class_array
        self.true_class  = true_class_array

    def explain(self, n_runs=10, explainer_lambda=0.8, save_to_disk=False):
        """
        Explain the model's results.
        """

        ############################################
        # Run the Explainer
        ############################################

        LOC = self.location
        model = self.model
        s2v_test_dataset = self.s2v_test_dataset
        dataset = self.dataset
        gene_names = self.gene_names

        if self.verbose:
            print("# Run the Explainer")

        no_of_runs = n_runs
        lamda = 0.8 # not used!
        ems = []
        NODE_MASK = list()

        for idx in range(no_of_runs):
            if self.verbose >= 1:
                print(f'Explainer::Iteration {idx+1} of {no_of_runs}')
            exp = GNNExplainer(model, epochs=300)
            em = exp.explain_graph_modified_s2v(s2v_test_dataset, lamda)
            #Path(f"{path}/{sigma}/modified_gnn").mkdir(parents=True, exist_ok=True)
            gnn_feature_masks = np.reshape(em, (len(em), -1))
            NODE_MASK.append(np.array(gnn_feature_masks.sigmoid()))
            np.savetxt(f'{LOC}/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            #np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            gnn_edge_masks = calc_edge_importance(gnn_feature_masks, dataset[0].edge_index)
            np.savetxt(f'{LOC}/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            #np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            ems.append(gnn_edge_masks.sigmoid().numpy())

        ems     = np.array(ems)
        mean_em = ems.mean(0)

        # OUTPUT -- Save Edge Masks
        np.savetxt(f'{LOC}/edge_masks.txt', mean_em, delimiter=',', fmt='%.5f')
        self.edge_mask = mean_em
        self.node_mask_matrix = np.concatenate(NODE_MASK,1)
        self.node_mask = np.concatenate(NODE_MASK,1).mean(1)

        self._explainer_run = True

        ###############################################
        # Perform Community Detection
        ###############################################

        avg_mask, coms = find_communities(f'{LOC}/edge_index.txt', f'{LOC}/edge_masks.txt')
        self.modules = coms
        self.module_importances = avg_mask

        np.savetxt(f'{LOC}/communities_scores.txt', avg_mask, delimiter=',', fmt='%.3f')

        filePath = f'{LOC}/communities.txt'

        if os.path.exists(filePath):
            os.remove(filePath)

        f = open(f'{LOC}/communities.txt', "a")
        for idx in range(len(avg_mask)):
            s_com = ','.join(str(e) for e in coms[idx])
            f.write(s_com + '\n')

        f.close()

        # Write gene_names to file
        textfile = open(f'{LOC}/gene_names.txt', "w")
        for element in gene_names:
            listToStr = ''.join(map(str, element))
            textfile.write(listToStr + "\n")

        textfile.close()

        self._explainer_run = True

    def predict(self, gnnsubnet_test):

        confusion_array = []
        true_class_array = []
        predicted_class_array = []

        s2v_test_dataset  = convert_to_s2vgraph(gnnsubnet_test.dataset)
        model = self.model
        model.eval()
        output = pass_data_iteratively(model, s2v_test_dataset)
        predicted_class = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
        correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
        acc_test = correct / float(len(s2v_test_dataset))

        #if use_weights:
        #    loss = nn.CrossEntropyLoss(weight=weight)(output,labels)
        #else:
        #    loss = nn.CrossEntropyLoss()(output,labels)
        #test_loss = loss

        predicted_class_array = np.append(predicted_class_array, predicted_class)
        true_class_array = np.append(true_class_array, labels)

        confusion_matrix_gnn = confusion_matrix(true_class_array, predicted_class_array)
        if self.verbose >= 1:
            print("## Confusion matrix:")
            print("## %s" % str(confusion_matrix_gnn))

        counter = 0
        for it, i in zip(predicted_class_array, range(len(predicted_class_array))):
            if it == true_class_array[i]:
                counter += 1

        accuracy = counter/len(true_class_array) * 100
        if self.verbose >= 1:
            print("## Accuracy: {}%".format(accuracy))

        self.predictions_test = predicted_class_array
        self.true_class_test  = true_class_array
        self.accuracy_test = accuracy
        self.confusion_matrix_test = confusion_matrix_gnn
        return(predicted_class_array)

    def download_TCGA(self, save_to_disk=False) -> None:
        """
        Warning: Currently not implemented!

        Download some sample TCGA data. Running this function will download
        approximately 100MB of data.
        """
        base_url = 'https://raw.githubusercontent.com/pievos101/GNN-SubNet/python-package/TCGA/' # CHANGE THIS URL WHEN BRANCH MERGES TO MAIN

        KIDNEY_RANDOM_Methy_FEATURES_filename = 'KIDNEY_RANDOM_Methy_FEATURES.txt'
        KIDNEY_RANDOM_PPI_filename = 'KIDNEY_RANDOM_PPI.txt'
        KIDNEY_RANDOM_TARGET_filename = 'KIDNEY_RANDOM_TARGET.txt'
        KIDNEY_RANDOM_mRNA_FEATURES_filename = 'KIDNEY_RANDOM_mRNA_FEATURES.txt'

        # For testing let's use KIDNEY_RANDOM_Methy_FEATURES and store in memory.
        raw = requests.get(base_url + KIDNEY_RANDOM_Methy_FEATURES_filename, stream=True)

        self.KIDNEY_RANDOM_Methy_FEATURES = np.asarray(pd.read_csv(io.BytesIO(raw.content), delimiter=' '))

        # Clear some memory
        raw = None

        return None

    def __len__(self) -> int:
        try:
            return self.edges.shape[0]
        except:
            return None

# GNNSubNet.py
# Authors: Bastian Pfeifer <https://github.com/pievos101>, Marcus D. Bloice <https://github.com/mdbloice>
from urllib.parse import _NetlocResultMixinStr
import numpy as np
import random
# from scipy.sparse.extract import find
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
# from collections.abc import Mapping

from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

from .gnn_training_utils import check_if_graph_is_connected, pass_data_iteratively
from .dataset import generate, load_OMICS_dataset, convert_to_s2vgraph
from .gnn_explainer import GNNExplainer
from .graphcnn import GraphCNN
from .graphcheb import GraphCheb, ChebConvNet, test_model_acc, test_model

from .community_detection import find_communities
from .edge_importance import calc_edge_importance

from torch_geometric.nn.conv.cheb_conv import ChebConv


class GNNSubNet(object):
    """
    The class GNNSubSet represents the main user API for the
    GNN-SubNet package.
    """

    def __init__(self, location=None, ppi=None, features=None, target=None, cutoff=950, normalize=True) -> None:

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

        # Flags for internal use (hidden from user)
        self._explainer_run = False

        if ppi == None:
            return None

        dataset, gene_names = load_OMICS_dataset(self.ppi, self.features, self.target, True, cutoff, normalize)

        # Check whether graph is connected
        check = check_if_graph_is_connected(dataset[0].edge_index)
        print("Graph is connected ", check)

        if check == False:
            print("Calculate subgraph ...")
            dataset, gene_names = load_OMICS_dataset(self.ppi, self.features, self.target, False, cutoff, normalize)

        check = check_if_graph_is_connected(dataset[0].edge_index)
        print("Graph is connected ", check)

        # print('\n')
        print('##################')
        print("# DATASET LOADED #")
        print('##################')
        # print('\n')

        self.dataset = dataset
        self.true_class = None
        self.gene_names = gene_names
        self.s2v_test_dataset = None
        self.edges = np.transpose(np.array(dataset[0].edge_index.detach().cpu().numpy))

        self.edge_mask = None
        self.node_mask = None
        self.node_mask_matrix = None
        self.modules = None
        self.module_importances = None

    def summary(self):
        """
        Print a summary for the GNNSubSet object's current state.
        """
        print("")
        print("Number of nodes:", len(self.dataset[0].x))
        print("Number of edges:", self.edges.shape[0])
        print("Number of modalities:", self.dataset[0].x.shape[1])

    def train(self, epoch_nr=25, method="graphcheb"):

        if method == "chebconv":
            print("chebconv for training ...")
            self.train_chebconv(epoch_nr=epoch_nr)
            self.classifier = "chebconv"

        if method == "graphcnn":
            print("graphcnn for training ...")
            self.train_graphcnn(epoch_nr=epoch_nr)
            self.classifier = "graphcnn"

        if method == "graphcheb":
            print("graphcheb for training ...")
            self.train_graphcheb(epoch_nr=epoch_nr)
            self.classifier = "graphcheb"

    def explain(self, n_runs=1, classifier="chebconv"):

        if self.classifier == "chebconv":
            self.explain_chebconv(n_runs=n_runs)

        if self.classifier == "graphcnn":
            self.explain_graphcnn(n_runs=n_runs)

        if self.classifier == "graphcheb":
            self.explain_graphcheb(n_runs=n_runs)

    def predict(self, gnnsubnet_test):

        if self.classifier == "chebconv":
            pred = self.predict_chebconv(gnnsubnet_test=gnnsubnet_test)

        if self.classifier == "graphcnn":
            pred = self.predict_graphcnn(gnnsubnet_test=gnnsubnet_test)

        if self.classifier == "graphcheb":
            pred = self.predict_graphcheb(gnnsubnet_test=gnnsubnet_test)

        pred = np.array(pred)
        pred = pred.reshape(1, pred.size)

        return pred

    def train_graphcheb(self, epoch_nr=25, shuffle=True, weights=False,
                        hidden_channels=10,
                        K=10,
                        layers_nr=1,
                        num_classes=2):
        """
        ---
        """
        use_weights = False

        dataset = self.dataset
        gene_names = self.gene_names

        graphs_class_0_list = []
        graphs_class_1_list = []
        for graph in dataset:
            if graph.y.detach().cpu().numpy() == 0:
                graphs_class_0_list.append(graph)
            else:
                graphs_class_1_list.append(graph)

        graphs_class_0_len = len(graphs_class_0_list)
        graphs_class_1_len = len(graphs_class_1_list)

        print(f"Graphs class 0: {graphs_class_0_len}, Graphs class 1: {graphs_class_1_len}")

        ########################################################################################################################
        # Downsampling of the class that contains more elements ===========================================================
        # ########################################################################################################################

        if graphs_class_0_len >= graphs_class_1_len:
            random_graphs_class_0_list = random.sample(graphs_class_0_list, graphs_class_1_len)
            balanced_dataset_list = graphs_class_1_list + random_graphs_class_0_list

        if graphs_class_0_len < graphs_class_1_len:
            random_graphs_class_1_list = random.sample(graphs_class_1_list, graphs_class_0_len)
            balanced_dataset_list = graphs_class_0_list + random_graphs_class_1_list

        # print(len(random_graphs_class_0_list))
        # print(len(random_graphs_class_1_list))

        random.shuffle(balanced_dataset_list)
        print(f"Length of balanced dataset list: {len(balanced_dataset_list)}")

        list_len = len(balanced_dataset_list)
        # print(list_len)
        train_set_len = int(list_len * 4 / 5)
        train_dataset_list = balanced_dataset_list[:train_set_len]
        test_dataset_list = balanced_dataset_list[train_set_len:]

        train_graph_class_0_nr = 0
        train_graph_class_1_nr = 0
        for graph in train_dataset_list:
            if graph.y.detach().cpu().numpy() == 0:
                train_graph_class_0_nr += 1
            else:
                train_graph_class_1_nr += 1
        print(f"Train graph class 0: {train_graph_class_0_nr}, train graph class 1: {train_graph_class_1_nr}")

        test_graph_class_0_nr = 0
        test_graph_class_1_nr = 0
        for graph in test_dataset_list:
            if graph.y.detach().cpu().numpy() == 0:
                test_graph_class_0_nr += 1
            else:
                test_graph_class_1_nr += 1
        print(f"Validation graph class 0: {test_graph_class_0_nr}, validation graph class 1: {test_graph_class_1_nr}")

        # s2v_train_dataset = convert_to_s2vgraph(train_dataset_list)
        # s2v_test_dataset  = convert_to_s2vgraph(test_dataset_list)
        s2v_train_dataset = train_dataset_list
        s2v_test_dataset = test_dataset_list

        model_path = 'omics_model.pth'
        no_of_features = dataset[0].x.shape[1]
        nodes_per_graph_nr = dataset[0].x.shape[0]
        print("\tnodes_per_graph_nr", nodes_per_graph_nr)

        input_dim = no_of_features
        n_classes = 2

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ChebConvNet(input_channels=1, n_features=nodes_per_graph_nr, n_channels=2, n_classes=2, K=8, n_layers=1)
        print(model)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9,
        #                                                       last_epoch=-1)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        min_loss = 50

        best_model = ChebConvNet(input_channels=1, n_features=nodes_per_graph_nr, n_channels=2, n_classes=2, K=8, n_layers=1)
        best_model.to(device)
        # best_model = ChebConv(in_channels=1, out_channels=2, K=10)

        min_val_loss = 1000000.0
        n_epochs_stop = 25
        epochs_no_improve = 0
        batch_size = 100

        train_loader = DataLoader(s2v_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(s2v_test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epoch_nr):
            running_loss = 0.0
            steps = 0
            model.train()
            # data_pbar_loader = tqdm(train_loader, unit='batch')
            for data in train_loader:
                out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
                loss = criterion(out, data.y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                running_loss += loss.item()
                steps += 1

            epoch_loss = running_loss / steps
            model.eval()
            acc_train = test_model_acc(train_loader, model)
            val_loss, val_acc, _ , _ = test_model(test_loader, model, criterion)

            print()
            print(f'Epoch: {epoch:03d}, Train Loss: {epoch_loss:.4f}, Train Acc: {acc_train:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}', end='\t')
            # print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            # print(f"Train Acc {acc_train:.4f}")

            # data_pbar_loader.set_description('epoch: %d' % (epoch))
            # val_loss = 0
            #
            # tr = DataLoader(s2v_test_dataset, batch_size=len(s2v_test_dataset), shuffle=False)
            # for vv in tr:
            #     # print("\toutput test")
            #     output = model(vv.x, vv.edge_index, vv.batch)
            #     # print("\toutput", output)
            #
            # # output = pass_data_iteratively(model, s2v_test_dataset)
            #
            # pred = output.max(1, keepdim=True)[1]
            # labels = torch.LongTensor([graph.y for graph in s2v_test_dataset])
            # if use_weights:
            #     loss = nn.CrossEntropyLoss(weight=weight)(output, labels)
            # else:
            #     loss = nn.CrossEntropyLoss()(output, labels)
            # val_loss += loss

            # print('Epoch {}, val_loss {:.4f}'.format(epoch, val_loss))
            if val_loss < min_val_loss and epoch > 2: # to go through at least 2 epochs
                print(f"Saving best model with validation loss {val_loss:.4f}", end="")
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    # model.load_state_dict(best_model.state_dict())
                    break
        #
        # confusion_array = []
        # true_class_array = []
        # predicted_class_array = []
        # model.eval()
        # correct = 0
        # true_class_array = []
        # predicted_class_array = []

        # loading the parameters of the best model
        model.load_state_dict(best_model.state_dict())

        _, _, true_labels, predicted_labels = test_model(test_loader, model, criterion)


        # test_loss = 0
        #
        # model.load_state_dict(best_model.state_dict())
        #
        # tr = DataLoader(s2v_test_dataset, batch_size=len(s2v_test_dataset), shuffle=False)
        # for vv in tr:
        #     output = model(vv.x, vv.edge_index, vv.batch)
        #
        # # output = pass_data_iteratively(model, s2v_test_dataset)
        # output = np.array(output.detach())
        # predicted_class = output.argmax(1, keepdims=True)
        #
        # predicted_class = list(predicted_class)
        #
        # labels = torch.LongTensor([graph.y for graph in s2v_test_dataset])
        # correct = torch.tensor(np.array(predicted_class)).eq(
        #     labels.view_as(torch.tensor(np.array(predicted_class)))).sum().item()



        confusion_matrix_gnn = confusion_matrix(true_labels, predicted_labels)
        print("\nConfusion matrix (Validation set):\n")
        print(confusion_matrix_gnn)

        from sklearn.metrics import balanced_accuracy_score
        acc_bal = balanced_accuracy_score(true_labels, predicted_labels)

        print("Validation balanced accuracy: {}".format(acc_bal))

        model.train()

        self.model_status = 'Trained'
        self.model = copy.deepcopy(model)
        self.accuracy = acc_bal
        self.confusion_matrix = confusion_matrix_gnn
        # self.test_loss = test_loss
        self.s2v_test_dataset = s2v_test_dataset
        self.predictions = predicted_labels
        self.true_class = true_labels

    def train_chebconv(self, epoch_nr=20, shuffle=True, weights=False):
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

        print(f"Graphs class 0: {graphs_class_0_len}, Graphs class 1: {graphs_class_1_len}")

        ########################################################################################################################
        # Downsampling of the class that contains more elements ===========================================================
        # ########################################################################################################################

        if graphs_class_0_len >= graphs_class_1_len:
            random_graphs_class_0_list = random.sample(graphs_class_0_list, graphs_class_1_len)
            balanced_dataset_list = graphs_class_1_list + random_graphs_class_0_list

        if graphs_class_0_len < graphs_class_1_len:
            random_graphs_class_1_list = random.sample(graphs_class_1_list, graphs_class_0_len)
            balanced_dataset_list = graphs_class_0_list + random_graphs_class_1_list

        random.shuffle(balanced_dataset_list)
        print(f"Length of balanced dataset list: {len(balanced_dataset_list)}")

        list_len = len(balanced_dataset_list)
        # print(list_len)
        train_set_len = int(list_len * 4 / 5)
        train_dataset_list = balanced_dataset_list[:train_set_len]
        test_dataset_list = balanced_dataset_list[train_set_len:]

        train_graph_class_0_nr = 0
        train_graph_class_1_nr = 0
        for graph in train_dataset_list:
            if graph.y.numpy() == 0:
                train_graph_class_0_nr += 1
            else:
                train_graph_class_1_nr += 1
        print(f"Train graph class 0: {train_graph_class_0_nr}, train graph class 1: {train_graph_class_1_nr}")

        test_graph_class_0_nr = 0
        test_graph_class_1_nr = 0
        for graph in test_dataset_list:
            if graph.y.numpy() == 0:
                test_graph_class_0_nr += 1
            else:
                test_graph_class_1_nr += 1
        print(f"Validation graph class 0: {test_graph_class_0_nr}, validation graph class 1: {test_graph_class_1_nr}")

        # for ChebConv()
        s2v_train_dataset = train_dataset_list
        s2v_test_dataset = test_dataset_list

        model_path = 'omics_model.pth'
        no_of_features = dataset[0].x.shape[1]
        nodes_per_graph_nr = dataset[0].x.shape[0]

        input_dim = no_of_features
        n_classes = 2

        # model = GraphCNN(num_layers, num_mlp_layers, input_dim, 32, n_classes, 0.5, True, graph_pooling_type, neighbor_pooling_type, 0)
        model = ChebConv(input_dim, n_classes, 10)

        opt = torch.optim.Adam(model.parameters(), lr=0.1)

        load_model = False
        if load_model:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            opt = checkpoint['optimizer']

        model.train()

        # min_loss = 50000
        # best_model = GraphCNN(num_layers, num_mlp_layers, input_dim, 32, n_classes, 0.5, True, graph_pooling_type, neighbor_pooling_type, 0)
        best_model = ChebConv(input_dim, n_classes, 10)

        min_val_loss = 1000000
        n_epochs_stop = 7
        epochs_no_improve = 0
        steps_per_epoch = 35

        for epoch in range(epoch_nr):
            model.train()
            pbar = tqdm(range(steps_per_epoch), unit='batch')
            epoch_loss = 0
            for pos in pbar:

                selected_idx = np.random.permutation(len(s2v_train_dataset))[:30]
                batch_graph = [s2v_train_dataset[idx] for idx in selected_idx]

                logits = []
                for g in batch_graph:
                    print('\t x and edge index')
                    print(type(g.x))

                    print(type(g.edge_index))
                    logits.append(model(x=g.x, edge_index=g.edge_index).max(0)[0])
                    # logits.append(model(x=g.x, edge_index=g.edge_index).mean(0))

                logits = torch.reshape(torch.cat(logits, 0), (30, 2))

                labels = torch.LongTensor([graph.y for graph in batch_graph])

                if use_weights:
                    loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
                else:
                    loss = nn.CrossEntropyLoss()(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.detach().item()

            epoch_loss /= steps_per_epoch
            model.eval()

            output = []
            for graphs in s2v_train_dataset:
                output.append(model(x=graphs.x, edge_index=graphs.edge_index).max(0)[0])
                # output.append(model(x=graphs.x, edge_index=graphs.edge_index).mean(0))

            output = torch.reshape(torch.cat(output, 0), (len(output), 2))

            output = np.array(output.detach())

            predicted_class = output.argmax(1, keepdims=True)

            predicted_class = list(predicted_class)

            labels = torch.LongTensor([graph.y for graph in s2v_train_dataset])

            correct = torch.tensor(np.array(predicted_class)).eq(
                labels.view_as(torch.tensor(np.array(predicted_class)))).sum().item()

            acc_train = correct / len(s2v_train_dataset)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            print(f"Train Acc {acc_train:.4f}")

            pbar.set_description('epoch: %d' % (epoch))
            val_loss = 0

            output = []
            for graphs in s2v_test_dataset:
                output.append(model(x=graphs.x, edge_index=graphs.edge_index).max(0)[0])
                # output.append(model(x=graphs.x, edge_index=graphs.edge_index).mean(0))

            output = torch.reshape(torch.cat(output, 0), (len(output), 2))

            labels = torch.LongTensor([graph.y for graph in s2v_test_dataset])

            if use_weights:
                loss = nn.CrossEntropyLoss(weight=weight)(output, labels)
            else:
                loss = nn.CrossEntropyLoss()(output, labels)
            val_loss += loss

            print('Epoch {}, val_loss {:.4f}'.format(epoch, val_loss))
            if val_loss < min_val_loss:
                print(f"Saving best model with validation loss {val_loss}")
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
                min_val_loss = val_loss

            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    model.load_state_dict(best_model.state_dict())
                    break

        confusion_array = []
        true_class_array = []
        predicted_class_array = []
        model.eval()
        correct = 0
        true_class_array = []
        predicted_class_array = []

        test_loss = 0

        model.load_state_dict(best_model.state_dict())

        output = []
        for graphs in s2v_test_dataset:
            output.append(model(x=graphs.x, edge_index=graphs.edge_index).max(0)[0])
            # output.append(model(x=graphs.x, edge_index=graphs.edge_index).mean(0))

        output = torch.reshape(torch.cat(output, 0), (len(output), 2))
        output = np.array(output.detach())
        predicted_class = output.argmax(1, keepdims=True)

        predicted_class = list(predicted_class)

        labels = torch.LongTensor([graph.y for graph in s2v_test_dataset])

        correct = torch.tensor(np.array(predicted_class)).eq(
            labels.view_as(torch.tensor(np.array(predicted_class)))).sum().item()

        confusion_matrix_gnn = confusion_matrix(labels, predicted_class)
        print("\nConfusion matrix (Validation set):\n")
        print(confusion_matrix_gnn)

        from sklearn.metrics import balanced_accuracy_score
        acc_bal = balanced_accuracy_score(labels, predicted_class)

        print("Validation accuracy: {}".format(acc_bal))

        model.train()

        self.model_status = 'Trained'
        self.model = copy.deepcopy(model)
        self.accuracy = acc_bal
        self.confusion_matrix = confusion_matrix_gnn
        # self.test_loss = test_loss
        self.s2v_test_dataset = s2v_test_dataset
        self.predictions = predicted_class_array
        self.true_class = labels

    def train_graphcnn(self, num_layers=1, num_mlp_layers=1, epoch_nr=10, shuffle=True, weights=False,
                       graph_pooling_type='sum1', neighbor_pooling_type='max'):
        """
        Train the GNN model on the data provided during initialisation.
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
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

        print(f"Graphs class 0: {graphs_class_0_len}, Graphs class 1: {graphs_class_1_len}")

        ########################################################################################################################
        # Downsampling of the class that contains more elements ===========================================================
        # ########################################################################################################################

        if graphs_class_0_len >= graphs_class_1_len:
            random_graphs_class_0_list = random.sample(graphs_class_0_list, graphs_class_1_len)
            balanced_dataset_list = graphs_class_1_list + random_graphs_class_0_list

        if graphs_class_0_len < graphs_class_1_len:
            random_graphs_class_1_list = random.sample(graphs_class_1_list, graphs_class_0_len)
            balanced_dataset_list = graphs_class_0_list + random_graphs_class_1_list

        # print(len(random_graphs_class_0_list))
        # print(len(random_graphs_class_1_list))

        random.shuffle(balanced_dataset_list)
        print(f"Length of balanced dataset list: {len(balanced_dataset_list)}")

        list_len = len(balanced_dataset_list)
        # print(list_len)
        train_set_len = int(list_len * 4 / 5)
        train_dataset_list = balanced_dataset_list[:train_set_len]
        test_dataset_list = balanced_dataset_list[train_set_len:]

        train_graph_class_0_nr = 0
        train_graph_class_1_nr = 0
        for graph in train_dataset_list:
            if graph.y.numpy() == 0:
                train_graph_class_0_nr += 1
            else:
                train_graph_class_1_nr += 1
        print(f"Train graph class 0: {train_graph_class_0_nr}, train graph class 1: {train_graph_class_1_nr}")

        test_graph_class_0_nr = 0
        test_graph_class_1_nr = 0
        for graph in test_dataset_list:
            if graph.y.numpy() == 0:
                test_graph_class_0_nr += 1
            else:
                test_graph_class_1_nr += 1
        print(f"Validation graph class 0: {test_graph_class_0_nr}, validation graph class 1: {test_graph_class_1_nr}")

        s2v_train_dataset = convert_to_s2vgraph(train_dataset_list)
        s2v_test_dataset = convert_to_s2vgraph(test_dataset_list)

        # TRAIN GNN -------------------------------------------------- #
        # count = 0
        # for item in dataset:
        #    count += item.y.item()

        # weight = torch.tensor([count/len(dataset), 1-count/len(dataset)])
        # print(count/len(dataset), 1-count/len(dataset))

        model_path = 'omics_model.pth'
        no_of_features = dataset[0].x.shape[1]
        nodes_per_graph_nr = dataset[0].x.shape[0]

        # print(len(dataset), len(dataset)*0.2)
        # s2v_dataset = convert_to_s2vgraph(dataset)
        # train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=123)
        # s2v_train_dataset = convert_to_s2vgraph(train_dataset)
        # s2v_test_dataset = convert_to_s2vgraph(test_dataset)
        # s2v_train_dataset, s2v_test_dataset = train_test_split(s2v_dataset, test_size=0.2, random_state=123)

        input_dim = no_of_features
        n_classes = 2

        model = GraphCNN(num_layers, num_mlp_layers, input_dim, 32, n_classes, 0.5, True, graph_pooling_type,
                         neighbor_pooling_type, 0)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)

        load_model = False
        if load_model:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            opt = checkpoint['optimizer']

        model.train()
        min_loss = 50
        best_model = GraphCNN(num_layers, num_mlp_layers, input_dim, 32, n_classes, 0.5, True, graph_pooling_type,
                              neighbor_pooling_type, 0)
        min_val_loss = 1000000
        n_epochs_stop = 5
        epochs_no_improve = 0
        steps_per_epoch = 35

        for epoch in range(epoch_nr):
            model.train()
            pbar = tqdm(range(steps_per_epoch), unit='batch')
            epoch_loss = 0
            for pos in pbar:
                selected_idx = np.random.permutation(len(s2v_train_dataset))[:32]

                batch_graph = [s2v_train_dataset[idx] for idx in selected_idx]
                logits = model(batch_graph)
                labels = torch.LongTensor([graph.label for graph in batch_graph])
                if use_weights:
                    loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
                else:
                    loss = nn.CrossEntropyLoss()(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.detach().item()

            epoch_loss /= steps_per_epoch
            model.eval()
            output = pass_data_iteratively(model, s2v_train_dataset)
            predicted_class = output.max(1, keepdim=True)[1]
            labels = torch.LongTensor([graph.label for graph in s2v_train_dataset])
            correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
            acc_train = correct / float(len(s2v_train_dataset))
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            print(f"Train Acc {acc_train:.4f}")

            pbar.set_description('epoch: %d' % (epoch))
            val_loss = 0
            output = pass_data_iteratively(model, s2v_test_dataset)

            pred = output.max(1, keepdim=True)[1]
            labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
            if use_weights:
                loss = nn.CrossEntropyLoss(weight=weight)(output, labels)
            else:
                loss = nn.CrossEntropyLoss()(output, labels)
            val_loss += loss

            print('Epoch {}, val_loss {:.4f}'.format(epoch, val_loss))
            if val_loss < min_val_loss:
                print(f"Saving best model with validation loss {val_loss}")
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
                min_val_loss = val_loss

            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    model.load_state_dict(best_model.state_dict())
                    break

        confusion_array = []
        true_class_array = []
        predicted_class_array = []
        model.eval()
        correct = 0
        true_class_array = []
        predicted_class_array = []

        test_loss = 0

        model.load_state_dict(best_model.state_dict())

        output = pass_data_iteratively(model, s2v_test_dataset)
        predicted_class = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
        correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
        acc_test = correct / float(len(s2v_test_dataset))

        if use_weights:
            loss = nn.CrossEntropyLoss(weight=weight)(output, labels)
        else:
            loss = nn.CrossEntropyLoss()(output, labels)
        test_loss = loss

        predicted_class_array = np.append(predicted_class_array, predicted_class)
        true_class_array = np.append(true_class_array, labels)

        confusion_matrix_gnn = confusion_matrix(true_class_array, predicted_class_array)
        print("\nConfusion matrix (Validation set):\n")
        print(confusion_matrix_gnn)

        counter = 0
        for it, i in zip(predicted_class_array, range(len(predicted_class_array))):
            if it == true_class_array[i]:
                counter += 1

        accuracy = counter / len(true_class_array) * 100
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
        self.true_class = true_class_array

    def explain_graphcheb(self, n_runs=10, explainer_lambda=0.8, save_to_disk=False):
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

        print("")
        print("------- Run the Explainer -------")
        print("")

        no_of_runs = n_runs
        lamda = 0.8  # not used!
        ems = []
        NODE_MASK = list()

        for idx in range(no_of_runs):
            print(f'Explainer::Iteration {idx + 1} of {no_of_runs}')
            exp = GNNExplainer(model, epochs=300)
            em = exp.explain_graph_modified_cheb2(s2v_test_dataset, lamda)
            # Path(f"{path}/{sigma}/modified_gnn").mkdir(parents=True, exist_ok=True)
            gnn_feature_masks = np.reshape(em.cpu(), (len(em), -1))
            NODE_MASK.append(np.array(gnn_feature_masks.sigmoid()))
            np.savetxt(f'{LOC}/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            # np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            gnn_edge_masks = calc_edge_importance(gnn_feature_masks, dataset[0].edge_index)
            np.savetxt(f'{LOC}/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            # np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            ems.append(gnn_edge_masks.sigmoid().numpy())

        ems = np.array(ems)
        mean_em = ems.mean(0)

        # OUTPUT -- Save Edge Masks
        np.savetxt(f'{LOC}/edge_masks.txt', mean_em, delimiter=',', fmt='%.5f')
        self.edge_mask = mean_em
        self.node_mask_matrix = np.concatenate(NODE_MASK, 1)
        self.node_mask = np.concatenate(NODE_MASK, 1).mean(1)

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

    def explain_chebconv(self, n_runs=10, explainer_lambda=0.8, save_to_disk=False):
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

        print("")
        print("------- Run the Explainer -------")
        print("")

        no_of_runs = n_runs
        lamda = 0.8  # not used!
        ems = []
        NODE_MASK = list()

        for idx in range(no_of_runs):
            print(f'Explainer::Iteration {idx + 1} of {no_of_runs}')
            exp = GNNExplainer(model, epochs=300)
            em = exp.explain_graph_modified_cheb(s2v_test_dataset, lamda)
            # Path(f"{path}/{sigma}/modified_gnn").mkdir(parents=True, exist_ok=True)
            gnn_feature_masks = np.reshape(em, (len(em), -1))
            NODE_MASK.append(np.array(gnn_feature_masks.sigmoid()))
            np.savetxt(f'{LOC}/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            # np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            gnn_edge_masks = calc_edge_importance(gnn_feature_masks, dataset[0].edge_index)
            np.savetxt(f'{LOC}/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            # np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            ems.append(gnn_edge_masks.sigmoid().numpy())

        ems = np.array(ems)
        mean_em = ems.mean(0)

        # OUTPUT -- Save Edge Masks
        np.savetxt(f'{LOC}/edge_masks.txt', mean_em, delimiter=',', fmt='%.5f')
        self.edge_mask = mean_em
        self.node_mask_matrix = np.concatenate(NODE_MASK, 1)
        self.node_mask = np.concatenate(NODE_MASK, 1).mean(1)

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

    def explain_graphcnn(self, n_runs=10, explainer_lambda=0.8, save_to_disk=False):
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

        print("")
        print("------- Run the Explainer -------")
        print("")

        no_of_runs = n_runs
        lamda = 0.8  # not used!
        ems = []
        NODE_MASK = list()

        for idx in range(no_of_runs):
            print(f'Explainer::Iteration {idx + 1} of {no_of_runs}')
            exp = GNNExplainer(model, epochs=300)
            em = exp.explain_graph_modified_s2v(s2v_test_dataset, lamda)
            # Path(f"{path}/{sigma}/modified_gnn").mkdir(parents=True, exist_ok=True)
            gnn_feature_masks = np.reshape(em, (len(em), -1))
            NODE_MASK.append(np.array(gnn_feature_masks.sigmoid()))
            np.savetxt(f'{LOC}/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            # np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_feature_masks{idx}.csv', gnn_feature_masks.sigmoid(), delimiter=',', fmt='%.3f')
            gnn_edge_masks = calc_edge_importance(gnn_feature_masks, dataset[0].edge_index)
            np.savetxt(f'{LOC}/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            # np.savetxt(f'{path}/{sigma}/modified_gnn/gnn_edge_masks{idx}.csv', gnn_edge_masks.sigmoid(), delimiter=',', fmt='%.3f')
            ems.append(gnn_edge_masks.sigmoid().numpy())

        ems = np.array(ems)
        mean_em = ems.mean(0)

        # OUTPUT -- Save Edge Masks
        np.savetxt(f'{LOC}/edge_masks.txt', mean_em, delimiter=',', fmt='%.5f')
        self.edge_mask = mean_em
        self.node_mask_matrix = np.concatenate(NODE_MASK, 1)
        self.node_mask = np.concatenate(NODE_MASK, 1).mean(1)

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

    def predict_graphcheb(self, gnnsubnet_test):

        s2v_test_dataset = gnnsubnet_test.dataset
        nodes_per_graph_nr = s2v_test_dataset[0].x.shape[0]
        # print("\nPredict_graphcheb function")
        # print("Nodes_per_graph_nr", nodes_per_graph_nr)

        model = self.model
        model.eval()

        test_loader = DataLoader(s2v_test_dataset, batch_size=len(s2v_test_dataset), shuffle=False)
        _, _, true_labels, predicted_labels = test_model(test_loader, model)

        confusion_matrix_gnn = confusion_matrix(true_labels, predicted_labels)
        print("\nConfusion matrix (using predict_graphcheb_function):")
        print(confusion_matrix_gnn)

        from sklearn.metrics import balanced_accuracy_score
        acc_bal = balanced_accuracy_score(true_labels, predicted_labels)

        # print("Validation balanced accuracy (predict_graphcheb): {}".format(acc_bal))
        print(f'Nodes_number: {nodes_per_graph_nr:03d}, Validation balanced accuracy (using predict_graphcheb): {acc_bal:.4f}')

        self.predictions_test = predicted_labels
        self.true_class_test = true_labels
        self.accuracy_test = acc_bal
        self.confusion_matrix_test = confusion_matrix_gnn

        return predicted_labels

    def predict_chebconv(self, gnnsubnet_test):

        confusion_array = []
        true_class_array = []
        predicted_class_array = []

        s2v_test_dataset = gnnsubnet_test.dataset
        model = self.model
        model.eval()

        output = []
        for graphs in s2v_test_dataset:
            output.append(model(x=graphs.x, edge_index=graphs.edge_index).max(0)[0])
            # output.append(model(x=graphs.x, edge_index=graphs.edge_index).mean(0))

        output = torch.reshape(torch.cat(output, 0), (len(output), 2))
        output = np.array(output.detach())
        predicted_class = output.argmax(1, keepdims=True)

        predicted_class = list(predicted_class)

        labels = torch.LongTensor([graph.y for graph in s2v_test_dataset])

        correct = torch.tensor(np.array(predicted_class)).eq(
            labels.view_as(torch.tensor(np.array(predicted_class)))).sum().item()

        confusion_matrix_gnn = confusion_matrix(labels, predicted_class)
        print("\nConfusion matrix (Validation set):\n")
        print(confusion_matrix_gnn)

        from sklearn.metrics import balanced_accuracy_score
        acc_bal = balanced_accuracy_score(labels, predicted_class)

        print("Validation accuracy: {}".format(acc_bal))

        self.predictions_test = predicted_class
        self.true_class_test = labels
        self.accuracy_test = acc_bal
        self.confusion_matrix_test = confusion_matrix_gnn

        return predicted_class

    def predict_graphcnn(self, gnnsubnet_test):

        confusion_array = []
        true_class_array = []
        predicted_class_array = []

        s2v_test_dataset = convert_to_s2vgraph(gnnsubnet_test.dataset)
        model = self.model
        model.eval()
        output = pass_data_iteratively(model, s2v_test_dataset)
        predicted_class = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in s2v_test_dataset])
        correct = predicted_class.eq(labels.view_as(predicted_class)).sum().item()
        acc_test = correct / float(len(s2v_test_dataset))

        # if use_weights:
        #    loss = nn.CrossEntropyLoss(weight=weight)(output,labels)
        # else:
        #    loss = nn.CrossEntropyLoss()(output,labels)
        # test_loss = loss

        predicted_class_array = np.append(predicted_class_array, predicted_class)
        true_class_array = np.append(true_class_array, labels)

        confusion_matrix_gnn = confusion_matrix(true_class_array, predicted_class_array)
        print("\nConfusion matrix:\n")
        print(confusion_matrix_gnn)

        counter = 0
        for it, i in zip(predicted_class_array, range(len(predicted_class_array))):
            if it == true_class_array[i]:
                counter += 1

        accuracy = counter / len(true_class_array) * 100
        print("Accuracy: {}%".format(accuracy))

        self.predictions_test = predicted_class_array
        self.true_class_test = true_class_array
        self.accuracy_test = accuracy
        self.confusion_matrix_test = confusion_matrix_gnn

        return predicted_class_array

    def download_TCGA(self, save_to_disk=False) -> None:
        """
        Warning: Currently not implemented!

        Download some sample TCGA data. Running this function will download
        approximately 100MB of data.
        """
        base_url = 'https://raw.githubusercontent.com/pievos101/GNN-SubNet/python-package/TCGA/'  # CHANGE THIS URL WHEN BRANCH MERGES TO MAIN

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

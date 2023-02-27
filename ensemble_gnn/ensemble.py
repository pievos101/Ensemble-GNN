from torch_geometric.data.data import Data
import torch
import copy
import numpy as np
import random

class ensemble(object):
    """
    The class ensemble represents the main user API for the
    Ensemble_GNN package.
    """
    def __init__(self, gnnsubnet=None, method="chebconv", epoch_nr: int = 20, niter: int = 1, verbose: int = 0) -> None:

        self.id = None
        self.ensemble  = list()
        self.train_data = None
        self.test_data  = None
        self.target = None
        self.train_accuracy = list()
        self.test_accuracy = None
        self.verbose = verbose

        if gnnsubnet == None:
            return None

        # store the data from the graph
        self.dataset = gnnsubnet.dataset
        self.gene_names = gnnsubnet.gene_names
        self.modules_gene_names = list()

        # train
        gnnsubnet.train(epoch_nr=epoch_nr, method=method)

        # explainer
        gnnsubnet.explain(n_runs=niter, classifier=method)

        # get subnets and build ensemble classifier
        self.modules = gnnsubnet.modules
        self.module_importances = gnnsubnet.module_importances

        if self.verbose:
            print("# Building the Ensembles #")

        # build the ensemble from these modules
        ## split the data sets
        for xx in range(len(self.modules)):
            if self.verbose >=1:
                print("## Ensemble:: %d of %d" %(xx+1, len(self.modules)) )
            self.ensemble.append(copy.deepcopy(gnnsubnet))
            #print(len(self.ensemble))
            mod_sub = gnnsubnet.modules[xx]
            # now cut data to module
            data_c = copy.deepcopy(gnnsubnet.dataset[0])
            #print(data_c.edge_index)
            x       = data_c.x
            x_sub   = x[mod_sub]
            eindex  = data_c.edge_index
            eindex0 = data_c.edge_index[0,]
            eindex1 = data_c.edge_index[1,]
            res = []
            i = 0
            while (i < len(eindex0)):
                if (mod_sub.count(eindex0[i]) > 0):
                    res.append(i)
                i += 1
            eindex0_sub = res
            res = []
            i = 0
            while (i < len(eindex1)):
                if (mod_sub.count(eindex1[i]) > 0):
                    res.append(i)
                i += 1
            eindex1_sub = res

            ids = np.intersect1d(eindex0_sub, eindex1_sub)

            e0 = np.array(eindex[0,][ids])
            e1 = np.array(eindex[1,][ids])

            # convert ids
            res = []
            i = 0
            while (i < len(e0)):
                ids = np.where(e0[i]==np.array(mod_sub))[0]
                res.append(ids)
                i += 1
            e0_final = np.array(res).flatten()

            res = []
            i = 0
            while (i < len(e1)):
                ids = np.where(e1[i]==np.array(mod_sub))[0]
                res.append(ids)
                i += 1
            e1_final = np.array(res).flatten()

            #print(e0_final)
            #print(e1_final)

            graphs  = []
            for yy in range(len(gnnsubnet.dataset)):
                #print(f'Samples:: {yy+1} of {len(gnnsubnet.dataset)}')
                data_c  = copy.deepcopy(gnnsubnet.dataset[yy])
                x       = data_c.x.numpy()
                x_sub   = x[mod_sub]
                graphs.append(Data(x=torch.tensor(x_sub).float(),
                edge_index=torch.tensor([e0_final,e1_final], dtype=torch.long),
                            y=data_c.y))

            self.ensemble[xx].dataset = graphs
            self.ensemble[xx].gene_names = self.gene_names[mod_sub]
            self.modules_gene_names.append(self.gene_names[mod_sub])
            self.ensemble[xx].modules = mod_sub
            self.ensemble[xx].model = None
            self.ensemble[xx].classifier = method

    def add(self, gnnsubnet):

        self.gnnsubnet = self.ensemble.append(gnnsubnet)

    def train(self):

        train_data = self.train_data
        acc = list()

        for xx in range(len(self.ensemble)):
            self.ensemble[xx].train(method=self.ensemble[xx].classifier)
            acc.append(self.ensemble[xx].accuracy)

        self.train_accuracy = acc

    def grow(self, size=10):

        values = np.array(self.train_accuracy)
        ens  = list()
        acc  = list()
        p    = (values - values.min()) / (values - values.min()).sum()
        samp = np.random.multinomial(size, p , size=1)
        samp = samp[0]

        for xx in range(len(samp)):
            times = samp[xx]
            for yy in range(times):
                ens.append(copy.deepcopy(self.ensemble[xx]))
                acc.append(self.ensemble[xx].accuracy)
        self.ensemble = ens
        self.train_accuracy = acc

    def predict(self, gnnsubnet_test):
        acc  = list()
        pred = list()
        true_labels = list()
        for xx in range(len(self.ensemble)):
            copy_test = copy.deepcopy(gnnsubnet_test)
            testgraphs=[]
            for yy in range(len(copy_test.dataset)):
                testgraphs.append(Data(x=torch.tensor(np.array(copy_test.dataset[yy].x)[self.ensemble[xx].modules]).float(),
                    edge_index=torch.tensor(np.array(self.ensemble[xx].dataset[0].edge_index), dtype=torch.long),
                            y=copy_test.dataset[yy].y))
            copy_test.dataset = testgraphs
            self.ensemble[xx].predict(copy_test)
            acc.append(self.ensemble[xx].accuracy)
            pred.append(self.ensemble[xx].predictions_test)
            true_labels.append(self.ensemble[xx].true_class_test)
        self.accuracy_test = acc
        self.predictions_test = pred
        self.true_class_test  = true_labels

        # Majority Vote
        pred_mv = np.zeros(len(pred[0]))
        for xx in range(len(pred)):
            for yy in range(len(pred[0])):
                if pred[xx][yy] == 1:
                    pred_mv[yy] = pred_mv[yy] + 1
                if pred[xx][yy] == 0:
                    pred_mv[yy] = pred_mv[yy] - 1
        ids0 = np.where(pred_mv<=0)
        ids1 = np.where(pred_mv>0)
        pred_mv[ids0] = 0
        pred_mv[ids1] = 1
        self.predictions_test_mv = pred_mv
        return(pred_mv)

    def send_model(self):
        m  = list()
        for xx in range(len(self.ensemble)):
            gnn_c = copy.deepcopy(self.ensemble[xx])
            gnn_c.dataset = [gnn_c.dataset[0]]
            gnn_c.dataset[0].x = None
            gnn_c.dataset[0].y = None
            m.append(gnn_c)
        return m

    def propose(self):
        ens_len = len(self.ensemble)
        # randomly select a member
        id = random.randint(0, ens_len)
        acc = self.ensemble[id].accuracy
        dat = copy.deepcopy(self.ensemble[id].dataset)
        names = self.ensemble[id].gene_names

        return dat[0].edge_index, names, acc

    #def check(self, subnet):
        # check the subnet on the data

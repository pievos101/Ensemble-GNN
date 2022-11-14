from torch_geometric.data.data import Data
import torch
from copy import copy
import numpy as np

class ensemble(object):
    """
    The class ensemble represents the main user API for the
    Ensemble_GNN package.
    """
    def __init__(self, gnnsubnet, niter=1) -> None:

        self.id = None
        self.ensemble  = list()
        self.train_data = None
        self.test_data  = None
        self.target = None
        self.train_accuracy = list()
        self.test_accuracy = None
        
        # store the data from the graph
        self.dataset = gnnsubnet.dataset
        self.gene_names = gnnsubnet.gene_names
        self.modules_gene_names = list()

        # train
        gnnsubnet.train()

        # explainer
        gnnsubnet.explain(niter)

        # get subnets and build ensemble classifier
        self.modules = gnnsubnet.modules
        self.module_importances = gnnsubnet.module_importances

        print('')
        print('##########################')
        print("# Building the Ensembles #")
        print('##########################')
        print('')
        
        # build the ensemble from these modules
        ## split the data sets
        for xx in range(len(self.modules)):
            print(f'Ensemble:: {xx+1} of {len(self.modules)}')
            graphs=[]
            self.ensemble.append(copy(gnnsubnet))
            #print(len(self.ensemble))
            mod_sub = gnnsubnet.modules[xx]
            # now cut data to module
            # ugly: the while loops can be moved out of the for loop
            for yy in range(len(gnnsubnet.dataset)):
                print(f'Samples:: {yy+1} of {len(gnnsubnet.dataset)}')
                data_c = copy(gnnsubnet.dataset[yy])
                #print(data_c.edge_index)
                x = data_c.x
                x_sub = x[mod_sub]
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

                import numpy as np
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

                graphs.append(Data(x=torch.tensor(x_sub).float(),
                edge_index=torch.tensor([e0_final,e1_final], dtype=torch.long),
                            y=data_c.y))

            self.ensemble[xx].dataset = graphs
            self.ensemble[xx].gene_names = self.gene_names[mod_sub]
            self.modules_gene_names.append(self.gene_names[mod_sub])


    def add(self, gnnsubnet):

        self.gnnsubnet = self.ensemble.append(gnnsubnet)

    def train(self):

        train_data = self.train_data
        acc = list()

        for xx in range(len(self.ensemble)):
            self.ensemble[xx].train()
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
                ens.append(copy(self.ensemble[xx]))
                acc.append(self.ensemble[xx].accuracy)
        self.ensemble = ens
        self.train_accuracy = acc

    def predict(self, gnnsubnet_test):

        acc  = list()
        pred = list()
        true_labels = list()
        for xx in range(len(self.ensemble)):
            self.ensemble[xx].predict(gnnsubnet_test)
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
from torch_geometric.data.data import Data
import torch
from copy import copy

class ensemble(object):
    """
    The class ensemble represents the main user API for the
    Ensemble_GNN package.
    """
    def __init__(self, gnnsubnet) -> None:

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
        gnnsubnet.explain(1)

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
            graphs=[]
            self.ensemble.append(copy(gnnsubnet))
            #print(len(self.ensemble))
            mod_sub = gnnsubnet.modules[xx]
            # now cut data to module
            for yy in range(len(gnnsubnet.dataset)):
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
                
    #def predict(self):
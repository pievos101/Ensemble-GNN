class ensemble(object):
    """
    The class ensemble represents the main user API for the
    Ensemble_GNN package.
    """
    def __init__(self, id) -> None:

        self.id = id
        self.ensemble  = list()
        self.train_data = None
        self.test_data  = None
        self.target = None
        self.train_accuracy = list()
        self.test_accuracy = None


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
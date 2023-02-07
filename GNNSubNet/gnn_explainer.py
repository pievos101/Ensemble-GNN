from copy import copy
from math import sqrt
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph, to_networkx

EPS = 1e-15


class GNNExplainer(torch.nn.Module):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.
    .. note::
        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.
    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        num_hops (int, optional): The number of hops the :obj:`model` is
            aggregating information from.
            If set to :obj:`None`, will automatically try to detect this
            information based on the number of
            :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`
            layers inside :obj:`model`. (default: :obj:`None`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model returns
            the logarithm of probabilities), :obj:`"prob"` (the model returns
            probabilities) and :obj:`"raw"` (the model returns raw scores).
            (default: :obj:`"log_prob"`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 log: bool = True):
        super(GNNExplainer, self).__init__()
        assert return_type in ['log_prob', 'prob', 'raw']
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.return_type = return_type
        self.log = log

    def __set_masks__(self, x, edge_index, init="normal", type=2):
        (N, F), E = x.size(), edge_index.size(1)

        #print(N)
        #print(F)    

        std = 0.1
        if type == 1: # mask the features
            #self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)
            self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)

        if type == 2: # mask the nodes
            #self.node_feat_mask = torch.nn.Parameter(torch.randn(N) * 0.1)
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, 1) * std)
        

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                print("Module is Instance of MessagePassing")
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def __loss__(self, node_idx, log_logits, pred_label):
        # node_idx is -1 for explaining graphs
        loss = -log_logits[
            node_idx, pred_label[node_idx]] if node_idx != -1 else -log_logits[
                0, pred_label[0]]

        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def explain_graph(self, data, **kwargs):
        r"""Learns and returns an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.
        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        x, edge_index = data.x, data.edge_index

        self.model.eval()
        self.__clear_masks__()

        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(data=data, batch=batch, **kwargs)
            log_logits = self.__to_log_prob__(out)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        #optimizer = torch.optim.Adam([self.edge_mask],
        #                             lr=self.lr)                        
                        

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            data_copy = copy(data)
            optimizer.zero_grad()
            h = x * self.node_feat_mask.view(1, -1).sigmoid()
            data_copy.x = h
            out = self.model(data=data_copy, batch=batch, **kwargs)
            log_logits = self.__to_log_prob__(out)
            loss = self.__loss__(-1, log_logits, pred_label)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if epoch%100==0:
                print(f"Epoch {epoch} Loss {loss.item()}")

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return node_feat_mask, edge_mask

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.
        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, mapping, hard_edge_mask, kwargs = self.__subgraph__(
            node_idx, x, edge_index, **kwargs)

        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, **kwargs)
            log_logits = self.__to_log_prob__(out)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.view(1, -1).sigmoid()
            out = self.model(x=h, edge_index=edge_index, **kwargs)
            log_logits = self.__to_log_prob__(out)
            loss = self.__loss__(mapping, log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs):
        r"""Visualizes the subgraph given an edge mask
        :attr:`edge_mask`.
        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`-1` to explain graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.
        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        import matplotlib.pyplot as plt
        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(
                edge_index.max() + 1,
                device=edge_index.device if y is None else y.device)
        else:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_kwargs = copy(kwargs)
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    
    def explain_graph_s2v(self, dataset, param, idd):
        # TODO -- vanilla GNNexplainer without sampling
        self.model.eval()
        self.__clear_masks__()    

        PRED = []
        LOGITS = []
        LOGITS2 =[]
        # Get the initial prediction.
        with torch.no_grad():
            #for yy in range(len(dataset)):
            x, edge_index = dataset[idd].node_features, dataset[idd].edge_mat
            out = self.model([dataset[idd]])
            log_logits = self.__to_log_prob__(out)
            pp = log_logits.argmax(dim=-1)
            PRED.append(pp)
            LOGITS.append(-log_logits[0, pp])
            LOGITS2.append(-log_logits[0, :])    

        self.__set_masks__(dataset[0].node_features,dataset[0].edge_mat)
        self.to(x.device)
        
        n_nodes = dataset[0].node_features.size()[0]

        optimizer = torch.optim.Adam([self.edge_mask, self.node_feat_mask], lr=self.lr)
        
        #optimizer = torch.optim.Adam([self.edge_mask],
        #                             lr=self.lr)                        
                        
        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
        
        for epoch in range(1, self.epochs + 1):
            #loss_xx  = 0 
            #sampSize = 10
            #if epoch%50==1: 
            #    ids  = np.random.randint(len(dataset), size=sampSize)
                
            optimizer.zero_grad()
            #for dd in ids: 
            data = dataset[idd]
            data_copy = copy(data)
            h = data.node_features * self.node_feat_mask.sigmoid()
            data_copy.node_features = h
            out = self.model([data_copy])
            log_logits = self.__to_log_prob__(out)
            loss_hit  = self.__loss__(-1, log_logits, PRED[0])
            loss_fail = self.__loss__(-1, log_logits, abs(PRED[0]-1))
            loss_xx = loss_hit 

        loss_xx.backward()
        optimizer.step()
         
        return self.node_feat_mask.view(-1,1).detach() #self.edge_mask.detach().sigmoid()


    def explain_graph_modified(self, dataset, param):
        self.model.eval()
        self.__clear_masks__()    

        PRED = []
        LOGITS = []
        LOGITS2 =[]
        # Get the initial prediction.
        with torch.no_grad():
            for yy in range(len(dataset)):
                x, edge_index = dataset[yy].x, dataset[yy].edge_index
                batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
                out = self.model(dataset[yy], batch=batch)
                log_logits = self.__to_log_prob__(out)
                pp = log_logits.argmax(dim=-1)
                PRED.append(pp)
                LOGITS.append(-log_logits[0, pp])
                LOGITS2.append(-log_logits[0, :])    

        self.__set_masks__(dataset[0].x,dataset[0].edge_index)
        self.to(x.device)
                                  
        optimizer = torch.optim.Adam([self.edge_mask, self.node_feat_mask], lr=self.lr)
       # optimizer = torch.optim.Adam([self.edge_mask],
       #                              lr=self.lr)                        
                        
        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
        
        for epoch in range(1, self.epochs + 1):
            loss_xx  = 0 
            sampSize = 10
            if epoch%50==1: 
                ids  = np.random.randint(len(dataset), size=sampSize)
                
            optimizer.zero_grad()
            for dd in ids: 
                data = dataset[dd]
                data_copy = copy(data)
                h = data.x * self.node_feat_mask.view(1, -1).sigmoid()
                data_copy.x = h
                #print(self.edge_mask.detach().sigmoid())
                out = self.model(data_copy, batch=batch)
                log_logits = self.__to_log_prob__(out)
                loss_hit  = self.__loss__(-1, log_logits, PRED[dd])
                loss_fail = self.__loss__(-1, log_logits, abs(PRED[dd]-1))
                #loss_xx = loss_xx + loss_hit + abs(LOGITS2[dd][PRED[dd]] - (-log_logits[0, PRED[dd][0]]))
                loss_xx = loss_xx + param*loss_hit + (1-param)*loss_fail
            #print(loss_xx)
            loss_xx.backward()
            optimizer.step()
            
        return self.edge_mask.detach().sigmoid()


    def explain_graph_modified_s2v(self, dataset, param):
        self.model.eval()
        self.__clear_masks__()    

        PRED = []
        LOGITS = []
        LOGITS2 =[]
        # Get the initial prediction.
        with torch.no_grad():
            for yy in range(len(dataset)):
                x, edge_index = dataset[yy].node_features, dataset[yy].edge_mat
                out = self.model([dataset[yy]])
                log_logits = self.__to_log_prob__(out)
                pp = log_logits.argmax(dim=-1)
                PRED.append(pp)
                LOGITS.append(-log_logits[0, pp])
                LOGITS2.append(-log_logits[0, :])    

        self.__set_masks__(dataset[0].node_features,dataset[0].edge_mat)
        self.to(x.device)
        
        n_nodes = dataset[0].node_features.size()[0]

        optimizer = torch.optim.Adam([self.edge_mask, self.node_feat_mask], lr=self.lr)
        
        #optimizer = torch.optim.Adam([self.edge_mask],
        #                             lr=self.lr)                                  

        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
        
        for epoch in range(1, self.epochs + 1):
            loss_xx  = 0 
            sampSize = 10
            if epoch%50==1: 
                ids  = np.random.randint(len(dataset), size=sampSize)
                
            optimizer.zero_grad()
            for dd in ids: 
                data = dataset[dd]
                data_copy = copy(data)
                h = data.node_features * self.node_feat_mask.sigmoid()
                data_copy.node_features = h
                out = self.model([data_copy])
                log_logits = self.__to_log_prob__(out)
                loss_hit  = self.__loss__(-1, log_logits, PRED[dd])
                loss_fail = self.__loss__(-1, log_logits, abs(PRED[dd]-1))
                loss_xx = loss_xx + loss_hit 
            loss_xx.backward()
            optimizer.step()
         
        return self.node_feat_mask.view(-1,1).detach() #self.edge_mask.detach().sigmoid()

    def explain_graph_modified_s2v_API(self, dataset, param, node_mask=False):

        self.model.eval()
        self.__clear_masks__()    

        PRED = []
        LOGITS = []
        LOGITS2 =[]
        # Get the initial prediction.
        with torch.no_grad():
            for yy in range(len(dataset)):
                x, edge_index = dataset[yy].node_features, dataset[yy].edge_mat
                out = self.model([dataset[yy]])
                log_logits = self.__to_log_prob__(out)
                pp = log_logits.argmax(dim=-1)
                PRED.append(pp)
                LOGITS.append(-log_logits[0, pp])
                LOGITS2.append(-log_logits[0, :])    

        if node_mask==False:
            self.__set_masks__(dataset[0].node_features, dataset[0].edge_mat)
        else:
            (N, F), E = dataset[0].node_features.size(), dataset[0].edge_mat.size(1)
            std = 0.1
            # inverse of sigmoid
            node_mask = np.log(1/(1-node_mask))
            # transform to tensor 
            node_mask_tensor = torch.nn.Parameter(torch.from_numpy(node_mask*std))
            self.node_feat_mask = torch.reshape(node_mask_tensor, (N, 1))
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        self.to(x.device)
        
        n_nodes = dataset[0].node_features.size()[0]

        optimizer = torch.optim.Adam([self.edge_mask, self.node_feat_mask], lr=self.lr)
        
        #optimizer = torch.optim.Adam([self.edge_mask],
        #                             lr=self.lr)                                  

        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
        
        for epoch in range(1, self.epochs + 1):
            loss_xx  = 0 
            sampSize = 10
            if epoch%50==1: 
                ids  = np.random.randint(len(dataset), size=sampSize)
                
            optimizer.zero_grad()
            for dd in ids: 
                data = dataset[dd]
                data_copy = copy(data)
                h = data.node_features * self.node_feat_mask.sigmoid()
                data_copy.node_features = h
                out = self.model([data_copy])
                log_logits = self.__to_log_prob__(out)
                loss_hit  = self.__loss__(-1, log_logits, PRED[dd])
                loss_fail = self.__loss__(-1, log_logits, abs(PRED[dd]-1))
                loss_xx = loss_xx + loss_hit 
            loss_xx.backward()
            optimizer.step()
         
        return self.node_feat_mask.view(-1,1).detach() #self.edge_mask.detach().sigmoid()




    def plot_graph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None,**kwargs):
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.
        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.
        :rtype: :class:`matplotlib.pyplot`
        """

        assert edge_mask.size(0) == edge_index.size(1)
        
        if threshold is not None:
            print('Edge Threshold:',threshold)
            edge_mask = (edge_mask >= threshold).to(torch.float)
          
        if node_idx is not None:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, hard_edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            edge_mask = edge_mask[hard_edge_mask]
        else:
            subset=[]
            for index,mask in enumerate(edge_mask):
                node_a = edge_index[0,index]
                node_b = edge_index[1,index]
                if node_a not in subset:
                    subset.append(node_a.cpu().item())
        #                     print("add: "+node_a)
                if node_b not in subset:
                    subset.append(node_b.cpu().item())
        #                     print("add: "+node_b)
        #             subset = torch.cat(subset).unique()
        edge_list=[]
        for index, edge in enumerate(edge_mask):
            if edge:
                edge_list.append((edge_index[0,index].cpu(),edge_index[1,index].cpu()))
        
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index.cpu(), att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')

        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        #         mapping = {k: i for k, i in enumerate(subset.tolist())}
        mapping = {k: i for k, i in enumerate(subset)}
        #         print(mapping)
        #         G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 200
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
        #                     connectionstyle="arc3,rad=0.1",
                ))
        # #         if node_feature_mask is not None:
        nx.draw_networkx_nodes(G, pos, **kwargs)

        color = np.array(edge_mask.cpu())

        nx.draw_networkx_edges(G, pos,
                       width=3, alpha=0.5, edge_color=color,edge_cmap=plt.cm.Reds)
        nx.draw_networkx_labels(G, pos, **kwargs)
        plt.axis('off')
        return plt
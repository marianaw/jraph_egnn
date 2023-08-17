import numpy as np
import jax
import jax.numpy as jnp
import jraph
import random
import networkx as nx


class BaseGraphDataset:

    def __init__(self, n_nodes, n_samples, partition, batch_size, key, K=8, directed=True):

        self.partition = partition
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.directed = directed
        self._batch_size = batch_size
        self._repeat = True
        self._key = key
        self.K = K
        
        if self.partition == 'train':
            self.seed = 0
            self.n_samples = 5000
        elif self.partition == 'val':
            self.seed = 1
            self.n_samples = 500
        elif self.partition == 'test':
            self.seed = 2
            self.n_samples = 500
        else:
            self.seed = 42

        self.n_samplesxnodes = int(self.n_samples / len(self.n_nodes))

    def _build(self):
        self._graphs, self._n_edges, self._n_nodes = self.create()
        self.total_num_graphs = len(self._graphs)

        self._generator = self._make_generator()
        self._max_nodes = int(np.max(self._n_nodes))
        self._max_edges = int(np.max(self._n_edges))
        # self._sum_nodes = int(np.sum(self._n_nodes))
        # self._sum_edges = int(np.sum(self._n_edges))
    
        self._generator = jraph.dynamically_batch(
            self._generator,
            n_node=self._batch_size * (self._max_nodes) + 1,
            # Times two because we want backwards edges.
            n_edge=self._batch_size * (self._max_edges),
            n_graph=self._batch_size + 1
        )

        # If n_node = [1,2,3], we create accumulated n_node [0,1,3,6] for indexing.
        self._accumulated_n_nodes = np.concatenate((np.array([0]),
                                                    np.cumsum(self._n_nodes)))
        # Same for n_edge
        self._accumulated_n_edges = np.concatenate((np.array([0]),
                                                    np.cumsum(self._n_edges)))

    def create_jgraph(self):
        raise NotImplementedError
    
    def _get_dense_edges(self, n_nodes, pos_senders, pos_receivers):
        edge_idx = jnp.arange(n_nodes**2)
        send = edge_idx // n_nodes
        rec = edge_idx % n_nodes

        if pos_senders.shape[0] == 0:
            labels = jnp.zeros((n_nodes**2, 1))
        else:
            idx = list(zip(pos_senders, pos_receivers))
            tgt = jnp.zeros((n_nodes, n_nodes))
            tgt = tgt.at[tuple(jnp.array(idx).T)].set(1.0)
            labels = tgt.reshape(-1, 1)

        self_idx = send == rec
        send = send[~self_idx]
        rec = rec[~self_idx]

        edge_attr = labels[~self_idx]
        return send, rec, edge_attr, labels

    def create(self):
        graphs, n_edges, n_nodes = [], [], []
        random.seed(self.seed)
        for i in range(self.n_samplesxnodes):
            for n in self.n_nodes:
                G, n_edge, n_node = self.create_jgraph(n)
                graphs.append(G)
                n_edges.append(n_edge)
                n_nodes.append(n)
        
        all = list(zip(graphs, n_edges, n_nodes))
        random.shuffle(all)
        graphs, n_edges, n_nodes = zip(*all)
        return graphs, np.array(n_edges), np.array(n_nodes)
    
    def _get_key(self):
        self._key, newkey = jax.random.split(self._key)
        return newkey

    def _make_generator(self):
        
        idx = 0
        while True:
            if not self._repeat:
                if idx == self.total_num_graphs:
                    return
            else:
                # This will reset the index to 0 if we are at the end of the dataset.
                idx = idx % self.total_num_graphs

            graph = self._graphs[idx]
            idx += 1
            yield graph

    def repeat(self):
        self._repeat = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)


class ErdosRenyDataset(BaseGraphDataset):

    def __init__(self, p, **kwargs):
        super().__init__(**kwargs)

        self.p = p
        self._build()

    def create_jgraph(self, n_nodes):
        G = nx.gnp_random_graph(n_nodes, self.p, directed=False)
        if self.directed:
            G = G.to_directed()
        n_edges = len(G.edges)

        if n_edges == 0:
            senders = [1]   # Just add one edge.
            receivers = [0]
        else:
            senders, receivers = zip(*G.edges)

        key = self._get_key()
        jG = jraph.GraphsTuple(
                n_node=jnp.array([n_nodes]), 
                n_edge=jnp.array([n_edges]),
                nodes={'nodes': jnp.ones((n_nodes, 1)),
                       'position': jax.random.normal(key, (n_nodes, self.K))
                },  # Add coordinates
                senders=jnp.array(senders), 
                receivers=jnp.array(receivers),
                edges=None,
                globals=None
            )

        return jG, n_edges, n_nodes


class DenseErdosRenyDataset(BaseGraphDataset):


    def __init__(self, p, **kwargs):
        '''
        Same as before but with all edges and edge type (1 or 0) as
        edge attributes. This is useful for doing link prediction and
        it's taken from the E(n)-GNN auto-encoder experimental 
        setting.
        '''
        super().__init__(**kwargs)

        self.p = p
        self._build()
    
    def _get_dense_edges(self, n_nodes, pos_senders, pos_receivers):
        edge_idx = jnp.arange(n_nodes**2)
        send = edge_idx // n_nodes
        rec = edge_idx % n_nodes

        if pos_senders.shape[0] == 0:
            labels = jnp.zeros(n_nodes**2).reshape(-1, 1)
        else:
            idx = list(zip(pos_senders, pos_receivers))
            tgt = jnp.zeros((n_nodes, n_nodes))
            tgt = tgt.at[tuple(jnp.array(idx).T)].set(1.0)
            labels = tgt.reshape(-1, 1)

        self_idx = send == rec
        send = send[~self_idx]
        rec = rec[~self_idx]

        edge_attr = labels[~self_idx]
        return send, rec, edge_attr, labels
    
    def create_jgraph(self, n_nodes):
        G = nx.gnp_random_graph(n_nodes, self.p, directed=False)
        if self.directed:
            G = G.to_directed()
        n_edges = len(G.edges)

        if n_edges == 0:
            pos_senders = []
            pos_receivers = []
        else:
            pos_senders, pos_receivers = zip(*G.edges)
        
        pos_senders = jnp.array(pos_senders)
        pos_receivers = jnp.array(pos_receivers)

        senders, receivers, edge_attr, labels = self._get_dense_edges(n_nodes, pos_senders, pos_receivers)

        key = self._get_key()
        jG = jraph.GraphsTuple(
                n_node=jnp.array([n_nodes]), 
                n_edge=jnp.array([n_edges]),
                nodes={'nodes': jnp.ones((n_nodes, 1)),
                       'position': jax.random.normal(key, (n_nodes, self.K))
                },  # Add coordinates
                senders=senders, 
                receivers=receivers,
                edges={'edges': edge_attr, 'labels': labels},
                globals=None
            )

        return jG, n_edges, n_nodes

    def __init__(self, p, **kwargs):
        '''
        Same as before but with all edges and edge type (1 or 0) as
        edge attributes. This is useful for doing link prediction and
        it's taken from the E(n)-GNN auto-encoder experimental 
        setting.
        '''
        super().__init__(**kwargs)

        self.p = p
        self._build()

    def create_jgraph(self, n_nodes):
        G = nx.gnp_random_graph(n_nodes, self.p, directed=False)
        if self.directed:
            G = G.to_directed()
        n_edges = len(G.edges)

        if n_edges == 0:
            pos_senders = []
            pos_receivers = []
        else:
            pos_senders, pos_receivers = zip(*G.edges)
        
        pos_senders = jnp.array(pos_senders)
        pos_receivers = jnp.array(pos_receivers)

        senders, receivers, edge_attr, labels = self._get_dense_edges(n_nodes, pos_senders, pos_receivers)

        key = self._get_key()
        jG = jraph.GraphsTuple(
                n_node=jnp.array([n_nodes]), 
                n_edge=jnp.array([n_edges]),
                nodes={'nodes': jnp.ones((n_nodes, 1)),
                       'position': jax.random.normal(key, (n_nodes, self.K))
                },  # Add coordinates
                senders=senders, 
                receivers=receivers,
                edges={'edges': edge_attr, 'labels': labels},
                globals=None
            )

        return jG, n_edges, n_nodes

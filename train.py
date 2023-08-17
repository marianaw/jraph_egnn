import pickle
import haiku as hk
import optax
import jax
import jax.numpy as jnp

from functools import partial
from tqdm import tqdm
from datasets import DenseErdosRenyDataset
from models import ae_egnn


def train(num_steps: int):
    n_nodes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    key = jax.random.PRNGKey(0)
    key, newkey = jax.random.split(key)
    train_dataset = DenseErdosRenyDataset(p=.25, partition='train', n_samples=0, batch_size=64, n_nodes=n_nodes, key=newkey)
    key, newkey = jax.random.split(key)
    test_dataset = DenseErdosRenyDataset(p=.25, partition='test', n_samples=0, batch_size=64, n_nodes=n_nodes, key=newkey)
    # random.seed(42)

    net_fn = ae_egnn
    net_fn = partial(net_fn, n_layers=n_layers)
    graph = train_dataset._graphs[0]
    network = hk.without_apply_rng(hk.transform(net_fn))

    key, newkey = jax.random.split(key)
    params = network.init(newkey, graph)

    def prediction_loss(params, graph):
        adj_pred = network.apply(params, graph)
        adj_pred = adj_pred.reshape(-1)
        tgt = graph.edges['edges'].reshape(-1)
        assert tgt.shape == adj_pred.shape
        loss = optax.sigmoid_binary_cross_entropy(adj_pred, tgt)
        loss = jnp.sum(loss)
        return loss
    
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(params)
    opt_update = optimizer.update

    compute_loss = jax.jit(jax.value_and_grad(prediction_loss, has_aux=False))
    
    @jax.jit
    def update(params, opt_state, graph):
        train_loss, g = compute_loss(params, graph)
        updates, opt_state = opt_update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, train_loss

    K = train_dataset.K
    main_pbar = tqdm(range(num_steps))
    ds_len = len(train_dataset._graphs)

    for step in main_pbar:
        key, newkey = jax.random.split(key)
        # ds_idx = jax.random.permutation(newkey, ds_len)
        chunk = train_dataset._graphs[:]
        pbar = tqdm(chunk, total=len(chunk), leave=False)
        ds_loss = 0
        for graph in pbar:  # We don't use batching, as in the original experiments.

            key, newkey = jax.random.split(key)
            nodes = graph.nodes['nodes']
            n_nodes = nodes.shape[0]
            position = jax.random.normal(newkey, (n_nodes, K))
            graph = graph._replace(nodes={'nodes': nodes, 'position': position})

            params, opt_state, train_loss = update(params, opt_state, graph)
            ds_loss += train_loss.item()

            b = params['element_wise_line']['b'].item()
            w = params['element_wise_line']['w'].item()
            pbar.set_description('Linear w={} and b={}'.format(w, b))

        ds_loss /= len(chunk)
        main_pbar.set_description('Train loss = {} at epoch {}'.format(ds_loss, step))

        if step % 20 == 0:
            test_loss = jnp.mean(
                jnp.asarray([
                    prediction_loss(params, test_dataset._graphs[i])
                    for i in range(100)
                ])).item()
            
            print("\nstep {} loss test {}\n".format(step, test_loss))
    
    return params


if __name__ == '__main__':
    n_layers = 4
    params = train(100)
    pickle.dump(params, open('model8.jax', 'wb'))

import haiku as hk
import jax
import jax.numpy as jnp
import jraph



class ElementWiseLine(hk.Module):

    def __call__(self, x):
        w_init = hk.initializers.Constant(-0.1)
        b_init = hk.initializers.Constant(1.0)
        w = hk.get_parameter('w', [], init=w_init)
        b = hk.get_parameter('b', [], init=b_init)
        return x * w + b


def egcn_layer(
    graph: jraph.GraphsTuple, recurrent: float = 1.0, coord_reg: float = 1e-3, i=1) -> jraph.GraphsTuple:

    def update_edge_fn(edges, senders, receivers, globals_):
        del globals_
        # import ipdb; ipdb.set_trace()
        coord_dif = senders["position"] - receivers["position"]
        # distance = jnp.linalg.norm(coord_dif, axis=1) ** 2
        distance = jnp.sum(coord_dif**2, -1, keepdims=True)
        edge_net = hk.Sequential(
            [hk.Linear(64, name='gcl_{}_edge_mlp_0'.format(i)), jax.nn.silu,
             hk.Linear(64, name='gcl_{}_edge_mlp_2'.format(i)), jax.nn.silu]
        )
        inputs = jnp.concatenate((receivers['nodes'], senders['nodes'], distance, edges['edges']), axis=1)
        out = edge_net(inputs)

        w_init = hk.initializers.VarianceScaling(scale=0.001, distribution='uniform', mode='fan_avg')
        coord_net = hk.Sequential(
            [hk.Linear(64, name='gcl_{}_coord_mlp_0'.format(i)), jax.nn.silu,
             hk.Linear(1, name= 'gcl_{}_coord_mlp_2'.format(i), with_bias=False, w_init=w_init)]
        )
        coord_out = coord_net(out)
        coord_emb = coord_out * coord_dif
        coord_emb = jnp.clip(coord_emb, -100, 100)
        return {'edges': out, 'coord_emb': coord_emb}

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        del sent_edges, globals_
        position = nodes['position']
        nodes = nodes['nodes']
        inputs = jnp.concatenate((nodes, received_edges['edges']), axis=1)
        node_net = hk.Sequential(
            [hk.Linear(64, name='gcl_{}_node_mlp_0'.format(i)), jax.nn.silu,
             hk.Linear(64, name='gcl_{}_node_mlp_2'.format(i))]
        )
        out = node_net(inputs) + recurrent * nodes
        agg = received_edges['coord_emb'] / (graph.n_node[0] -1)  # This works because it's a fully connected graph in this case.
        position = position + agg
        position = position - coord_reg * position
        
        return {"nodes": out, "position": position}

    gn = jraph.GraphNetwork(
          update_edge_fn=update_edge_fn,
          update_node_fn=update_node_fn,
          aggregate_edges_for_nodes_fn=jraph.segment_sum
    )
    return gn(graph)


vvd = lambda x, y: jnp.sum(x - y)**2  #  ([a], [a]) -> []
mvd = jax.vmap(vvd, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
mmd = jax.vmap(mvd, (None, 0), 0)


def ae_egnn(graph: jraph.GraphsTuple, n_layers: int = 4) -> jraph.ArrayTree:
    
    init_edges = graph.edges['edges'].astype(float)

    #Encode
    # import ipdb; ipdb.set_trace()
    for i in range(n_layers):
        recurrent = float(i > 0)
        graph = graph._replace(edges={'edges': init_edges})  # This is something E(n)GNN authors do.
        graph = egcn_layer(graph, recurrent=recurrent, i=i)

    coords = graph.nodes['position']
    sent = coords[graph.senders]
    rec = coords[graph.receivers]
    dists = sent - rec
    dists = jnp.sum(dists ** 2, -1)

    #Decode
    pred = ElementWiseLine()(dists)  # Unnormalized because the loss computes the sigmoid function
    # pred = jax.nn.sigmoid(ElementWiseLine()(dists))

    return pred

"""
This module contains multiple functions for clustering (networkx) graphs.
"""

import numpy as np
import networkx as nx
from collections import Counter

import chinese_whispers as cw
import community as community_louvain

""" Currently there is a problem with the graph-tools library,
    namely "libgraph_tool_core.so: undefined symbol: _ZN5boost6python15instance_holder8allocateEP7_objectmm"""
import graph_tool
from graph_tool.inference import minimize_blockmodel_dl
from graph_tool.inference.blockmodel import BlockState

from _correlation import *


def connected_components_clustering(
    graph: nx.Graph,
    positive_threshold: float = 0.0,
    is_non_value=lambda x: np.isnan(x),
) -> list:
    """Clusters the graph into connected components.

    Parameters
    ----------
    graph: networkx.Graph
        The graph for which to calculate the cluster labels
    positive_threshold: float
        The threshold for the positive weight. If the weight is below this value,
        the edge is considered as a negative edge.
    is_non_value: function
        A function that takes a value and returns True if the value is non-value.

    Returns
    -------
    classes : list[Set[int]]
        A list of sets of nodes, where each set is a cluster

    Raises
    ------
    ValueError
        If the graph contains non-value weights
    """
    if _check_nan_weights_exits(graph):
        raise ValueError(
            "NaN weights are not supported by the connected components method."
        )

    _graph = graph.copy()

    edges_negative = [
        (i, j)
        for (i, j) in _graph.edges()
        if _graph[i][j]["weight"] < positive_threshold
        or is_non_value(_graph[i][j]["weight"])
    ]
    _graph.remove_edges_from(edges_negative)
    components = nx.connected_components(_graph)

    classes = [set(component) for component in components]
    classes.sort(key=lambda x: len(x), reverse=True)

    return classes


def chinese_whispers_clustering(
    graph: nx.Graph,
    weighting: str = "top",
    iterations: int = 20,
    seed: int = None,
) -> list:
    """Cluster graph based on Chinese Whispers.

    Parameters
    ----------
    graph: networkx.Graph
        The graph for which to calculate the cluster labels
    weighting: str
        The weighting scheme to use. Can be 'top', 'lin' or 'log'
        - 'top': using weights directly
        - 'lin': normalize weights by the degree of the related node
        - 'log': normalize weights by the logarithm of the degree of the related node
    iterations: int
        The number of iterations to run the algorithm
    seed: int
        The seed for the random number generator

    Returns
    -------
    classes : list[Set[int]]
        A list of sets of nodes, where each set is a cluster

    Raises
    ------
    ValueError
        If the graph contains non-value weights
    """
    if _check_nan_weights_exits(graph):
        raise ValueError(
            "NaN weights are not supported by the Chinese Whispers method."
        )

    _graph = graph.copy()
    _cw_clustering = cw.aggregate_clusters(
        cw.chinese_whispers(
            _graph, weighting=weighting, iterations=iterations, seed=seed
        )
    )

    classes = [v for _, v in _cw_clustering.items()]
    classes.sort(key=lambda x: len(x), reverse=True)

    return classes


def louvain_clustering(
    graph: nx.Graph,
    init_partition: dict = None,
    resolution: float = 1.0,
    random_state=None,
) -> list:
    """Cluster graph based on Louvain Method.

    Parameters
    ----------
    graph: networkx.Graph
        The graph for which to calculate the cluster labels
    init_partition: dict
        A dictionary of node labels to cluster labels, which will be used as initial partition
    resolution: float
        The resolution parameter for the louvain method, see https://arxiv.org/pdf/0812.1770.pdf
    random_state: int, RandomState instance or None
        The random seed or state to use.


    Returns
    -------
    classes : list[Set[int]]
        A list of sets of nodes, where each set is a cluster

    Raises
    ------
    ValueError
        If the graph contains negative weights or non-value weights
    """

    if _negative_weights_exist(graph):
        raise ValueError("Negative weights are not supported by the Louvain method.")

    if _check_nan_weights_exits(graph):
        raise ValueError("NaN weights are not supported by the Louvain method.")

    _graph = graph.copy()

    _louvain_clustering = _invert_cluster_label_dict(
        community_louvain.best_partition(
            _graph,
            partition=init_partition,
            resolution=resolution,
            random_state=random_state,
        )
    )

    classes = [v for _, v in _louvain_clustering.items()]
    classes.sort(key=lambda x: len(x), reverse=True)

    return classes


def _invert_cluster_label_dict(cluster_label_dict: dict) -> dict:
    """Invert the cluster label dict.

    Parameters
    ----------
    cluster_label_dict: dict
        The cluster label dict to invert

    Returns
    -------
    inverted_cluster_label_dict: dict
        The inverted cluster label dict
    """
    inv_map = {}
    for k, v in cluster_label_dict.items():
        inv_map[v] = inv_map.get(v, set()).union({k})

    return inv_map


def wsbm_clustering(
    graph: nx.Graph,
    distribution: str = "discrete-binomial",
    use_disconnected_edges: bool = True,
) -> list:
    """Cluster graph based on Weighted Stochastic Block Model.

    Parameters
    ----------
    graph: networkx.Graph
        The graph for which to calculate the cluster labels
    distribution: str
        The distribution to use for the WSBM algorithm.
        Can be "real-exponential", "real-normal", "discrete-geometric", "discrete-poisson" or "discrete-binomial"

    Returns
    -------
    classes : list[Set[int]]
        A list of sets of nodes, where each set is a cluster

    Raises
    ------
    ValueError
        If the graph contains negative weights or non-value weights
    """

    if _negative_weights_exist(graph):
        raise ValueError("Negative weights are not supported by the WSBM algorithm.")

    if _check_nan_weights_exits(graph):
        raise ValueError("NaN weights are not supported by the WSBM algorithm.")

    gt_graph, _, gt2nx = _nxgraph_to_graphtoolgraph(
        graph.copy(), use_disconnected_edges=use_disconnected_edges
    )
    state: BlockState = _minimize(gt_graph, distribution)

    block2clusterid_map = {}
    for i, (k, _) in enumerate(
        dict(
            sorted(
                Counter(state.get_blocks().get_array()).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ).items()
    ):
        block2clusterid_map[k] = i

    communities = {}
    for i, block_id in enumerate(state.get_blocks().get_array()):
        nx_vertex_id = gt2nx[i]
        community_id = block2clusterid_map[block_id]
        if communities.get(community_id, None) is None:
            communities[community_id] = []
        communities[community_id].append(nx_vertex_id)

    classes = [set(v) for _, v in communities.items()]
    classes.sort(key=lambda x: len(x), reverse=True)

    return classes


def _nxgraph_to_graphtoolgraph(graph: nx.Graph, use_disconnected_edges: bool = True):
    """Convert a networkx graph to a graphtool graph.

    Parameters
    ----------
    graph: networkx.Graph
        The graph to convert
    use_disconnected_graph: bool
        Using the disconnected graphs with 0.0 as weights

    Returns
    -------
    gt_graph: graphtool.Graph
        The converted graph
    """
    graph_tool_graph = graph_tool.Graph(directed=False)

    nx2gt_vertex_id = dict()
    gt2nx_vertex_id = dict()
    for i, node in enumerate(graph.nodes()):
        nx2gt_vertex_id[node] = i
        gt2nx_vertex_id[i] = node

    new_weights = []
    for i, j in graph.edges():
        current_weight = graph[i][j]["weight"]
        if use_disconnected_edges is False:
            if current_weight != 0 and not np.isnan(current_weight):
                graph_tool_graph.add_edge(nx2gt_vertex_id[i], nx2gt_vertex_id[j])
                new_weights.append(current_weight)
        else:
            graph_tool_graph.add_edge(nx2gt_vertex_id[i], nx2gt_vertex_id[j])
            if current_weight == 0 or np.isnan(current_weight) == True:
                new_weights.append(0.0)
            else:
                new_weights.append(current_weight)

    original_edge_weights = graph_tool_graph.new_edge_property("double")
    original_edge_weights.a = new_weights
    graph_tool_graph.ep["weight"] = original_edge_weights

    new_vertex_id = graph_tool_graph.new_vertex_property("string")
    for k, v in nx2gt_vertex_id.items():
        new_vertex_id[v] = str(k)
    graph_tool_graph.vp.id = new_vertex_id

    return graph_tool_graph, nx2gt_vertex_id, gt2nx_vertex_id


def _minimize(graph: graph_tool.Graph, distribution: str) -> BlockState:
    """Minimize the graph using the given distribution as described by graph-tool.

    Parameters
    ----------
    graph: graphtool.Graph
        The graph to minimize
    distribution: str
        The distribution to use for the WSBM algorithm.

    Returns
    -------
    state: BlockState
        The minimized graph as BlockState object.
    """

    state = minimize_blockmodel_dl(
        graph,
        state_args=dict(
            deg_corr=False, recs=[graph.ep.weight], rec_types=[distribution]
        ),
        multilevel_mcmc_args=dict(
            B_min=1,
            B_max=30,
            # verbose=True,
            # niter=100,
            entropy_args=dict(adjacency=False, degree_dl=False),
        ),
    )

    for i in range(100):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=1)

    return state


def _negative_weights_exist(graph: nx.Graph):
    """Check if there are negative edges in the graph.

    Parameters
    ----------
    graph: networkx.Graph
        The graph to check negative edges for

    Returns
    -------
    flag: bool
        True if there are negative edges, False otherwise
    """
    for i, j in graph.edges():
        if graph[i][j]["weight"] < 0:
            return True
    return False


def _check_nan_weights_exits(graph: nx.Graph):
    """Check if there are NaN weights in the graph.

    Parameters
    ----------
    graph: networkx.Graph
        The graph to check NaN weights for

    Returns
    -------
    flag: bool
        True if there are NaN weights, False otherwise
    """
    for i, j in graph.edges():
        if np.isnan(graph[i][j]["weight"]):
            return True
    return False


def _adjacency_matrix_to_nxgraph(
    adjacency_matrix: np.ndarray, use_disconnected_edges: bool = True
) -> nx.Graph:
    """Convert an adjacency matrix to a networkx graph.

    Parameters
    ----------
    adjacency_matrix: np.ndarray
        The adjacency matrix to convert
    use_disconnected_edges: bool
        Used to add disconnected edges with 0.0 as weight

    Returns
    -------
    graph: networkx.Graph
        The converted graph
    """
    graph = nx.from_numpy_array(adjacency_matrix)
    if use_disconnected_edges is True:
        non_edges = list(nx.non_edges(graph))
        non_edges_weighted = [(u, v, 0.0) for u, v in non_edges]
        graph.add_weighted_edges_from(non_edges_weighted)
    return graph


def _convert_graph_cluster_list_set_to_list(
    graph: nx.Graph, cluster_list: list
) -> list:
    """Convert a list of clustered nodes to a list, where each .

    Parameters
    ----------
    classes : list[Set[int]]
        A list of sets of nodes, where each set is a cluster

    Returns
    -------
    cluster_list: list[int]
        The converted list
    """
    new_cluster_list = [0] * graph.number_of_nodes()

    for i, cluster in enumerate(cluster_list):
        for node in cluster:
            new_cluster_list[node] = i

    return new_cluster_list


def correlation_clustering(graph: nx.Graph, **params) -> list:
    if _check_nan_weights_exits(graph):
        raise ValueError(
            "NaN weights are not supported by the correlation clustering method."
        )
    params.pop("min_max", None)
    clusters, _ = cluster_correlation_search(graph, **params)
    return clusters


if __name__ == "__main__":
    # import numpy as np

    # A = np.array([[0, 1, 0], [2, 0, 0], [0, 0, 0]])

    # G = nx.from_numpy_matrix(A)

    # clusters = chinese_whispers_clustering(graph=G)

    # print(clusters)
    # print(
    #     _convert_graph_cluster_list_set_to_list(graph=G, cluster_list=clusters)
    # )

    graph = nx.Graph()
    A, B, C = "A", "B", "C"
    graph.add_edge(A, B, weight=0.0)
    graph.add_edge(A, C, weight=4)
    graph.add_edge(B, C, weight=np.nan)

    matrix = nx.adjacency_matrix(graph).todense()
    print(matrix)
    graph_converted = _adjacency_matrix_to_nxgraph(matrix)
    weights_graph_converted = {
        e: graph_converted.edges[e]["weight"] for e in graph_converted.edges
    }
    print(weights_graph_converted)
    (
        graph_tool_graph,
        nx2gt_vertex_id,
        gt2nx_vertex_id,
    ) = _nxgraph_to_graphtoolgraph(graph, True)
    print(list(graph_tool_graph.ep["weight"]))
    # non_edges = list(nx.non_edges(graph_converted))
    # print(non_edges)
    # non_edges_weighted = [(u, v, 0.0) for u, v in non_edges]
    # print(non_edges_weighted)
    # graph_converted.add_weighted_edges_from(non_edges_weighted)
    # print(graph_converted)
    # weights_graph_converted = {
    #     e: graph_converted.edges[e]["weight"] for e in graph_converted.edges
    # }
    # print(weights_graph_converted)
    # (
    #     graph_tool_graph,
    #     nx2gt_vertex_id,
    #     gt2nx_vertex_id,
    # ) = _nxgraph_to_graphtoolgraph(graph)
    # print(list(graph_tool_graph.ep["weight"]))

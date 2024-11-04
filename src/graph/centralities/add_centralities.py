import networkx as nx
import igraph as ig

from src.graph.intra_inter_graphs import separate_graph
from src.graph.centralities.hierarchical_measures import cal_k_core, cal_k_truss
from src.graph.centralities.betweenness_centrality import cal_betweenness_centrality
from src.graph.centralities.comm_centrality import comm_centrality
from src.graph.centralities.modularity_vitality import modularity_vitality


def add_centralities(df, new_path, graph_path, dataset, cn_measures, network_features):

    G = nx.from_pandas_edgelist(
        df, source=dataset.src_ip_col, target=dataset.dst_ip_col, create_using=nx.DiGraph())
    G.remove_nodes_from(list(nx.isolates(G)))
    for node in G.nodes():
        G.nodes[node]['label'] = node

    G1 = ig.Graph.from_networkx(G)
    labels = [G.nodes[node]['label'] for node in G.nodes()]
    G1.vs['label'] = labels

    part = G1.community_infomap()
    communities = []
    for com in part:
        communities.append([G1.vs[node_index]['label'] for node_index in com])

    community_labels = {}
    for i, community in enumerate(communities):
        for node in community:
            community_labels[node] = i

    nx.set_node_attributes(G, community_labels, "new_community")

    intra_graph, inter_graph = separate_graph(G, communities)

    if "betweenness" in cn_measures:
        nx.set_node_attributes(G, cal_betweenness_centrality(G), "betweenness")
        print("calculated betweenness")
    if "local_betweenness" in cn_measures:
        nx.set_node_attributes(G, cal_betweenness_centrality(
            intra_graph), "local_betweenness")
        print("calculated local_betweenness")
    if "global_betweenness" in cn_measures:
        nx.set_node_attributes(G, cal_betweenness_centrality(
            inter_graph), "global_betweenness")
        print("calculated global_betweenness")
    if "degree" in cn_measures:
        nx.set_node_attributes(G, nx.degree_centrality(G), "degree")
        print("calculated degree")
    if "local_degree" in cn_measures:
        nx.set_node_attributes(
            G, nx.degree_centrality(intra_graph), "local_degree")
        print("calculated local_degree")
    if "global_degree" in cn_measures:
        nx.set_node_attributes(G, nx.degree_centrality(
            inter_graph), "global_degree")
        print("calculated global_degree")
    if "eigenvector" in cn_measures:
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            G, max_iter=600), "eigenvector")
        print("calculated eigenvector")
    if "local_eigenvector" in cn_measures:
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            intra_graph), "local_eigenvector")
        print("calculated local_eigenvector")
    if "global_eigenvector" in cn_measures:
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            inter_graph), "global_eigenvector")
        print("calculated global_eigenvector")
    if "closeness" in cn_measures:
        nx.set_node_attributes(G, nx.closeness_centrality(G), "closeness")
        print("calculated closeness")
    if "local_closeness" in cn_measures:
        nx.set_node_attributes(G, nx.closeness_centrality(
            intra_graph), "local_closeness")
        print("calculated local_closeness")
    if "global_closeness" in cn_measures:
        nx.set_node_attributes(G, nx.closeness_centrality(
            inter_graph), "global_closeness")
        print("calculated global_closeness")
    if "pagerank" in cn_measures:
        nx.set_node_attributes(G, nx.pagerank(G, alpha=0.85), "pagerank")
        print("calculated pagerank")
    if "local_pagerank" in cn_measures:
        nx.set_node_attributes(G, nx.pagerank(
            intra_graph, alpha=0.85), "local_pagerank")
        print("calculated local_pagerank")
    if "global_pagerank" in cn_measures:
        nx.set_node_attributes(G, nx.pagerank(
            inter_graph, alpha=0.85), "global_pagerank")
        print("calculated global_pagerank")
    if "k_core" in cn_measures:
        nx.set_node_attributes(G, cal_k_core(G), "k_core")
        print("calculated k_core")
    if "k_truss" in cn_measures:
        nx.set_node_attributes(G, cal_k_truss(G), "k_truss")
        print("calculated k_truss")
    if "Comm" in cn_measures:
        nx.set_node_attributes(
            G, comm_centrality(G, community_labels), "Comm")
        print("calculated Comm")
    if "mv" in cn_measures:
        nx.set_node_attributes(G, modularity_vitality(G1, part), "mv")
        print("calculated mv")

    nx.write_gexf(G, graph_path)

    features_dicts = {}
    for measure in cn_measures:
        features_dicts[measure] = nx.get_node_attributes(G, measure)
        print(f"==>> features_dicts: {measure , len(features_dicts[measure])}")

    for feature in network_features:
        if feature[:3] == "src":
            df[feature] = df.apply(lambda row: features_dicts[feature[4:]].get(
                row[dataset.src_ip_col], -1), axis=1)
        if feature[:3] == "dst":
            df[feature] = df.apply(lambda row: features_dicts[feature[4:]].get(
                row[dataset.dst_ip_col], -1), axis=1)

    df.to_parquet(new_path)
    print(f"DataFrame written to {new_path}")

    return network_features

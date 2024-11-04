# import os
# import pandas as pd
# import networkx as nx
# import numpy as np
# from sklearn.model_selection import train_test_split

# if not os.path.isdir(FOLDER_PATH):
#     os.mkdir(FOLDER_PATH)


# def add_centralities(df, dataset):
#     G = nx.DiGraph()
#     for _, row in df.iterrows():
#         G.add_edge(row[dataset.src_ip_col], row[dataset.dst_ip_col])

#     degrees = nx.degree_centrality(G)
#     betwe = cal_betweenness_centrality(G)
#     pagerank = nx.pagerank(G, alpha=0.85)

#     df["src_degree"] = df.apply(lambda row: degrees.get(
#         row[dataset.src_ip_col], -1), axis=1)
#     df["dst_degree"] = df.apply(lambda row: degrees.get(
#         row[dataset.dst_ip_col], -1), axis=1)
#     df["src_betweenness"] = df.apply(
#         lambda row: betwe.get(row[dataset.src_ip_col], -1), axis=1)
#     df["dst_betweenness"] = df.apply(
#         lambda row: betwe.get(row[dataset.dst_ip_col], -1), axis=1)
#     df["src_pagerank"] = df.apply(lambda row: pagerank.get(
#         row[dataset.src_ip_col], -1), axis=1)
#     df["dst_pagerank"] = df.apply(lambda row: pagerank.get(
#         row[dataset.dst_ip_col], -1), axis=1)

#     return df


# def add_centralities_multidigraph(df, dataset):
#     G = nx.MultiDiGraph()
#     for _, row in df.iterrows():
#         G.add_edge(row[dataset.src_ip_col], row[dataset.dst_ip_col])

#     degrees = nx.degree_centrality(G)
#     betwe = cal_betweenness_centrality(G)
#     pagerank = nx.pagerank(G, alpha=0.85)

#     df["src_multidigraph_degree"] = df.apply(
#         lambda row: degrees.get(row[dataset.src_ip_col], -1), axis=1)
#     df["dst_multidigraph_degree"] = df.apply(
#         lambda row: degrees.get(row[dataset.dst_ip_col], -1), axis=1)
#     df["src_multidigraph_betweenness"] = df.apply(
#         lambda row: betwe.get(row[dataset.src_ip_col], -1), axis=1)
#     df["dst_multidigraph_betweenness"] = df.apply(
#         lambda row: betwe.get(row[dataset.dst_ip_col], -1), axis=1)
#     df["src_multidigraph_pagerank"] = df.apply(
#         lambda row: pagerank.get(row[dataset.src_ip_col], -1), axis=1)
#     df["dst_multidigraph_pagerank"] = df.apply(
#         lambda row: pagerank.get(row[dataset.dst_ip_col], -1), axis=1)

#     return df


# def process_datasets(df1, df2, class_col1, class_col2):
#     if not os.path.isdir(FOLDER_PATH):
#         os.mkdir(FOLDER_PATH)

#     train1, test1 = train_test_split(
#         df1, test_size=0.33, stratify=df1[class_col1])
#     train2, test2 = train_test_split(
#         df2, test_size=0.33, stratify=df2[class_col2])

#     test = pd.concat([test1, test2])
#     test = add_centralities(test, dataset1)
#     test = add_centralities_multidigraph(test, dataset1)
#     test.to_parquet(os.path.join(FOLDER_PATH, "test.parquet"))

#     client_data = np.array_split(train1, 5) + np.array_split(train2, 3)
#     for cid, data_partition in enumerate(client_data):
#         data_partition = add_centralities(data_partition, dataset1)
#         data_partition = add_centralities_multidigraph(
#             data_partition, dataset1)
#         data_partition.to_parquet(os.path.join(
#             FOLDER_PATH, f"client_{cid}.parquet"))


# if __name__ == "__main__":
#     process_datasets()

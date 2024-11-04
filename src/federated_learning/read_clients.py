import os
import pandas as pd
from sklearn.model_selection import train_test_split


def read_clients(folder_path, clients_paths, label_col, class_col, class_num_col, centralities_columns, pca_columns, drop_columns, weak_columns, multi_class):
    test = pd.read_parquet(folder_path + "test.parquet")

    if multi_class:
        test[label_col] = test[class_num_col]

    if centralities_columns:
        test.drop(centralities_columns[-1], axis=1, inplace=True)
    if pca_columns:
        test.drop(pca_columns, axis=1, inplace=True)

    test_by_class = {}
    classes = test[class_col].unique()
    for class_value in classes:
        test_class = test[test[class_col] == class_value].copy()
        test_class.drop(drop_columns, axis=1, inplace=True)
        test_class.drop(weak_columns, axis=1, inplace=True)
        test_class.reset_index(drop=True, inplace=True)

        test_class_labels = test_class[label_col].to_numpy()
        test_class = test_class.drop([label_col], axis=1).to_numpy()

        test_by_class[class_value] = (test_class, test_class_labels)

    test.drop(drop_columns, axis=1, inplace=True)
    test.drop(weak_columns, axis=1, inplace=True)
    test.reset_index(drop=True, inplace=True)

    test_labels = test[label_col].to_numpy()
    test = test.drop([label_col], axis=1).to_numpy()
    input_dim = test.shape[1]

    client_data = []
    for client_path in clients_paths:
        client_data.append(pd.read_parquet(client_path))

    for i in range(len(client_data)):

        cdata = client_data[i]

        if multi_class:
            cdata[label_col] = cdata[class_num_col]

        if centralities_columns:
            cdata.drop(centralities_columns[i], axis=1, inplace=True)
        if pca_columns:
            cdata.drop(pca_columns, axis=1, inplace=True)

        # cdata.drop(["src_degree", "dst_degree", "src_betweenness", "dst_betweenness", "src_pagerank", "dst_pagerank"], axis=1, inplace=True)
        # cdata.drop(["src_multidigraph_degree", "dst_multidigraph_degree", "src_multidigraph_betweenness", "dst_multidigraph_betweenness", "src_multidigraph_pagerank", "dst_multidigraph_pagerank"], axis=1, inplace=True)
        # cdata.drop(["pca_1", "pca_2"], axis=1, inplace=True)

        cdata.drop(drop_columns, axis=1, inplace=True)
        cdata.drop(weak_columns, axis=1, inplace=True)
        cdata.reset_index(drop=True, inplace=True)

        # Split into train, validation, and test sets
        c_train, c_test = train_test_split(cdata, test_size=0.1)

        # Split c_train further into c_train and c_val
        c_train, c_val = train_test_split(c_train, test_size=0.2)

        # Extract labels and features for train, validation, and test
        y_train = c_train[label_col].to_numpy()
        x_train = c_train.drop([label_col], axis=1).to_numpy()

        y_val = c_val[label_col].to_numpy()
        x_val = c_val.drop([label_col], axis=1).to_numpy()

        y_test = c_test[label_col].to_numpy()
        x_test = c_test.drop([label_col], axis=1).to_numpy()

        # Store in client_data: (x_train, y_train, x_val, y_val, x_test, y_test)
        client_data[i] = (x_train, y_train, x_val, y_val, x_test, y_test)

    return client_data, test, test_labels, test_by_class, input_dim

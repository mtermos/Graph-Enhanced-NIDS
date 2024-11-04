class DatasetInfo:
    def __init__(
            self,
            name,
            path,
            file_type,

            # Key Columns names
            src_ip_col,
            src_port_col,
            dst_ip_col,
            dst_port_col,
            flow_id_col,
            timestamp_col,
            label_col,
            class_col,

            class_num_col=None,
            timestamp_format=None,

            # Columns to be dropped from the dataset during preprocessing.
            drop_columns=[],

            # Result of Feature Analysis, Columns to be dropped from the dataset during preprocessing.
            weak_columns=[],
    ):

        self.name = name
        self.path = path
        self.file_type = file_type
        self.src_ip_col = src_ip_col
        self.src_port_col = src_port_col
        self.dst_ip_col = dst_ip_col
        self.dst_port_col = dst_port_col
        self.flow_id_col = flow_id_col
        self.timestamp_col = timestamp_col
        self.timestamp_format = timestamp_format
        self.label_col = label_col
        self.class_col = class_col
        self.class_num_col = class_num_col
        self.drop_columns = drop_columns
        self.weak_columns = weak_columns


datasets_list = [
    DatasetInfo(name="cic_ton_iot_5_percent",
                path="datasets/cic_ton_iot_5_percent/cic_ton_iot_5_percent.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="%d/%m/%Y %I:%M:%S %p",

                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Subflow Bwd Pkts', 'Flow IAT Mean', 'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Bwd Header Len', 'Tot Bwd Pkts', 'Bwd Pkt Len Mean', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
                              'CWE Flag Count', 'Bwd IAT Tot', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Min', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', 'Pkt Len Var', 'FIN Flag Cnt', 'Bwd IAT Mean', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Tot', 'PSH Flag Cnt', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_ids_2017_5_percent",
                path="datasets/cic_ids_2017_5_percent/cic_ids_2017_5_percent.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd IAT Min',  'Idle Max', 'Flow IAT Mean',  'Protocol',   'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Subflow Fwd Pkts', 'Bwd Pkt Len Mean', 'Tot Bwd Pkts', 'Pkt Size Avg',
                              'Subflow Bwd Pkts', 'Bwd IAT Std', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Std', 'Fwd URG Flags', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max',  'Pkt Len Var',  'Tot Fwd Pkts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd PSH Flags', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Max',  'Fwd IAT Tot', 'RST Flag Cnt', 'Subflow Bwd Byts', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
]


cn_measures_type_1 = ["betweenness", "local_betweenness", "degree", "local_degree",
                      "eigenvector", "closeness", "pagerank", "local_pagerank", "k_core", "k_truss", "Comm"]
network_features_type_1 = ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_degree', 'dst_degree', 'src_local_degree', 'dst_local_degree', 'src_eigenvector',
                           'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm']


# cn_measures_type_1 = ["betweenness", "local_betweenness", "degree", "local_degree",
#                       "closeness", "pagerank", "local_pagerank",  "k_truss", "Comm"]
# network_features_type_1 = ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_degree', 'dst_degree', 'src_local_degree', 'dst_local_degree',
#                            'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm']


cn_measures_type_2 = ["betweenness", "global_betweenness", "degree", "global_degree",
                      "eigenvector", "closeness", "pagerank", "global_pagerank", "k_core", "k_truss", "mv"]
network_features_type_2 = ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_degree', 'dst_degree', 'src_global_degree', 'dst_global_degree', 'src_eigenvector',
                           'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv']

cn_measures_type_3 = ["betweenness", "local_betweenness",
                      "pagerank", "local_pagerank", "k_core", "k_truss", "Comm"]
network_features_type_3 = ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_pagerank',
                           'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm']

cn_measures_type_4 = ["betweenness", "global_betweenness",
                      "pagerank", "global_pagerank", "k_core", "k_truss", "mv"]
network_features_type_4 = ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_pagerank',
                           'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv']


datasets = {dataset.name: dataset for dataset in datasets_list}

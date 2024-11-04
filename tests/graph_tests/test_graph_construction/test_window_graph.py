import unittest
import pandas as pd
import networkx as nx
import os
import shutil
from src.graph.graph_construction.window_graph import create_weightless_window_graph


class TestCreateWeightlessWindowGraph(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame
        self.df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-07-01 00:00:01', '2024-07-01 00:00:03', '2024-07-01 00:05:01',
                                         '2024-07-01 00:06:01', '2024-07-01 00:10:01']),
            'src_ip': ['192.168.1.1', '192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.1'],
            'dst_ip': ['192.168.2.1', '192.168.2.1', '192.168.2.1', '192.168.2.2', '192.168.2.1'],
            'src_port': [12345, 12345, 12345, 12346, 12345],
            'dst_port': [80, 80, 80, 80, 80],
            'protocol': ['TCP', 'TCP', 'TCP', 'TCP', 'TCP']
        })
        self.src_ip_col = 'src_ip'
        self.dst_ip_col = 'dst_ip'
        self.window_size = 2
        self.folder_path = 'test_graphs'

    def tearDown(self):
        # Clean up the test directory after each test
        if os.path.exists(self.folder_path):
            shutil.rmtree(self.folder_path)

    def test_create_weightless_window_graph(self):
        graphs = create_weightless_window_graph(self.df, self.src_ip_col, self.dst_ip_col,
                                                self.window_size, multi_graph=False, line_graph=False, folder_path=self.folder_path)
        # 5 records / window size of 2 = 3 windows
        self.assertEqual(len(graphs), 3)
        self.assertTrue(all(isinstance(graph, nx.DiGraph) for graph in graphs))
        self.assertTrue(os.path.exists(self.folder_path))
        self.assertEqual(len(os.listdir(self.folder_path)), 3)

    def test_create_weightless_multi_window_graph(self):
        graphs = create_weightless_window_graph(self.df, self.src_ip_col, self.dst_ip_col,
                                                self.window_size, multi_graph=True, line_graph=False, folder_path=self.folder_path)
        # 5 records / window size of 2 = 3 windows
        self.assertEqual(len(graphs), 3)
        self.assertTrue(all(isinstance(graph, nx.MultiDiGraph)
                        for graph in graphs))
        self.assertTrue(os.path.exists(self.folder_path))
        self.assertEqual(len(os.listdir(self.folder_path)), 3)


if __name__ == '__main__':
    unittest.main()

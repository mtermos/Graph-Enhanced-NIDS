import pandas as pd
import networkx as nx
import os
import shutil
import pytest
from graph.graph_construction.session_graph import define_sessions, create_weightless_session_graph


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-07-01 00:00:01', '2024-07-01 00:00:03', '2024-07-01 00:05:01',
                                     '2024-07-01 00:06:01', '2024-07-01 00:10:01']),
        'src_ip': ['192.168.1.1', '192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.1'],
        'dst_ip': ['192.168.2.1', '192.168.2.1', '192.168.2.1', '192.168.2.2', '192.168.2.1'],
        'src_port': [12345, 12345, 12345, 12346, 12345],
        'dst_port': [80, 80, 80, 80, 80],
        'protocol': ['TCP', 'TCP', 'TCP', 'TCP', 'TCP']
    })

# Cleans up the test directory after each test


@pytest.fixture
def folder_path():
    path = 'test_graphs'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    yield path
    shutil.rmtree(path)


def test_define_sessions(sample_data):
    src_ip_col = 'src_ip'
    dst_ip_col = 'dst_ip'
    src_port_col = 'src_port'
    dst_port_col = 'dst_port'
    protocol_col = 'protocol'
    timeout = pd.Timedelta(minutes=5)

    df_with_sessions = define_sessions(
        sample_data, src_ip_col, dst_ip_col, src_port_col, dst_port_col, protocol_col, timeout)
    assert 'session_id' in df_with_sessions.columns
    assert len(df_with_sessions['session_id'].unique()) == 2


def test_create_weightless_session_graph(sample_data, folder_path):
    src_ip_col = 'src_ip'
    dst_ip_col = 'dst_ip'
    src_port_col = 'src_port'
    dst_port_col = 'dst_port'
    protocol_col = 'protocol'
    timeout = pd.Timedelta(minutes=5)

    df_with_sessions = define_sessions(
        sample_data, src_ip_col, dst_ip_col, src_port_col, dst_port_col, protocol_col, timeout)
    graphs = create_weightless_session_graph(
        df_with_sessions, src_ip_col, dst_ip_col, multi_graph=False, line_graph=False, folder_path=folder_path)
    assert len(graphs) == 2
    assert all(isinstance(graph, nx.DiGraph) for graph in graphs)
    assert os.path.exists(folder_path)
    assert len(os.listdir(folder_path)) == 2


def test_create_weightless_multi_session_graph(sample_data, folder_path):
    src_ip_col = 'src_ip'
    dst_ip_col = 'dst_ip'
    src_port_col = 'src_port'
    dst_port_col = 'dst_port'
    protocol_col = 'protocol'
    timeout = pd.Timedelta(minutes=5)

    df_with_sessions = define_sessions(
        sample_data, src_ip_col, dst_ip_col, src_port_col, dst_port_col, protocol_col, timeout)
    graphs = create_weightless_session_graph(
        df_with_sessions, src_ip_col, dst_ip_col, multi_graph=True, line_graph=False, folder_path=folder_path)
    assert len(graphs) == 2
    assert all(isinstance(graph, nx.MultiDiGraph) for graph in graphs)
    assert os.path.exists(folder_path)
    assert len(os.listdir(folder_path)) == 2


if __name__ == '__main__':
    pytest.main()

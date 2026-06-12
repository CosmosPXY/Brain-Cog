"""
mul.py - 计算网络小世界特性指标

包含函数：
- mul_f1: 计算小世界系数（聚类系数和特征路径长度）
"""

import numpy as np
import networkx as nx
import os
from multiprocessing import Pool
import pickle


def calculate_small_world_metrics(graph_path):
    """
    计算单个网络的小世界指标
    
    参数：
        graph_path: 图的pickle文件路径
    
    返回：
        (clustering_coeff, characteristic_path_length): 聚类系数和特征路径长度
    """
    try:
        # 加载图
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        # 确保图非空
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return 0.0, float('inf')
        
        # 计算聚类系数（平均聚类系数）
        try:
            clustering_coeff = nx.average_clustering(G)
        except:
            clustering_coeff = 0.0
        
        # 计算特征路径长度
        # 对于连通的图分量，计算最大连通分量的平均最短路径
        try:
            if nx.is_connected(G):
                # 如果图是连通的，直接计算
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                # 如果图不连通，取最大连通分量
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                if len(subgraph) > 1:
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    avg_path_length = 1.0
        except:
            avg_path_length = float('inf')
        
        return clustering_coeff, avg_path_length
    
    except Exception as e:
        print(f"Error processing {graph_path}: {e}")
        return 0.0, float('inf')


def mul_f1(pop, steps=10, rootdir='./'):
    """
    计算种群中所有个体的小世界系数
    
    参数：
        pop: 种群大小（个体数量）
        steps: 时间步数（默认10，用于标准化）
        rootdir: 包含网络pickle文件的目录
    
    返回：
        results: 列表，每个元素为 (clustering_coeff, characteristic_path_length) 元组
    """
    
    results = []
    
    # 遍历每个个体对应的图文件
    for i in range(pop):
        graph_file = os.path.join(rootdir, f"{i}.pkl")
        
        if not os.path.exists(graph_file):
            print(f"Warning: Graph file not found: {graph_file}")
            results.append((0.0, 1.0))
            continue
        
        # 计算小世界指标
        clustering, path_length = calculate_small_world_metrics(graph_file)
        
        # 对指标进行归一化和处理
        # clustering_coeff: 范围 [0, 1]
        clustering = max(0.0, min(1.0, clustering))
        
        # characteristic_path_length: 避免无穷大
        if path_length == float('inf'):
            path_length = 1.0
        else:
            path_length = max(1.0, path_length)
        
        results.append((clustering, path_length))
    
    return results


def mul_f1_parallel(pop, steps=10, rootdir='./'):
    """
    使用多进程并行计算小世界系数（可选的并行版本）
    
    参数：
        pop: 种群大小
        steps: 时间步数
        rootdir: 包含网络pickle文件的目录
    
    返回：
        results: 列表，每个元素为 (clustering_coeff, characteristic_path_length) 元组
    """
    
    # 准备所有图文件路径
    graph_files = [os.path.join(rootdir, f"{i}.pkl") for i in range(pop)]
    
    # 使用多进程池计算
    with Pool() as pool:
        results = pool.map(calculate_small_world_metrics, graph_files)
    
    # 对结果进行后处理
    processed_results = []
    for clustering, path_length in results:
        clustering = max(0.0, min(1.0, clustering))
        path_length = 1.0 if path_length == float('inf') else max(1.0, path_length)
        processed_results.append((clustering, path_length))
    
    return processed_results


if __name__ == "__main__":
    # 测试代码
    print("mul.py - 小世界系数计算模块")
    print("用于ELSM进化算法")

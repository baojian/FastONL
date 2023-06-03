import os
import sys
import numpy as np
from algo import algo_weighted_tree
from utils import load_graph_data
import multiprocessing
import pickle as pkl

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def test_weighted_tree_simple():
    import networkx as nx
    edges = [(2, 7, 1.), (7, 2, 1.), (7, 8, 1. / 2.), (8, 7, 1. / 2.), (1, 7, 3.), (7, 1, 3.), (4, 7, 3.), (7, 4, 3.),
             (2, 6, 4.), (6, 2, 4.),
             (2, 9, 1. / 2.), (9, 2, 1. / 2.), (9, 5, 1. / 2.), (5, 9, 1. / 2.), (3, 9, 3.), (9, 3, 3.)]
    g = nx.Graph()
    for uu, vv, wei in edges:
        g.add_node(uu - 1)
        g.add_node(vv - 1)
        g.add_edge(uu - 1, vv - 1, weight=wei)
        g.add_edge(vv - 1, uu - 1, weight=wei)
    csr_mat = nx.to_scipy_sparse_array(g, nodelist=range(9))
    y = np.zeros(9)
    nodes = range(9)
    algo_weighted_tree(csr_mat, y, nodes, s=1, seed=1, verbose=True)


def run_weighted_tree_algorithm(para):
    dataset, trial_i, csr_mat, y, s, seed, nodes = para
    y_pred, rt = algo_weighted_tree(csr_mat, y, nodes, s, seed)
    return dataset, trial_i, rt, y, y_pred, nodes


def test_wta_small(num_cpus=1, num_trials=20, s=5):
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10', 'blogcatalog']
    weights = [False, False, False, False, True, False, False]
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        for trial_i in range(num_trials):
            np.random.seed(trial_i)
            nodes = np.random.permutation(n)
            para = (dataset, trial_i, csr_mat, y, s, trial_i, nodes)
            para_space.append(para)
    pool = multiprocessing.Pool(processes=num_cpus)
    pool_results = pool.map(func=run_weighted_tree_algorithm, iterable=para_space)
    pool.close()
    pool.join()
    results = dict()
    for dataset in dataset_list:
        results[dataset] = dict()
    for dataset, trial_i, rt, y, y_pred, nodes in pool_results:
        results[dataset][(trial_i, 'wta')] = rt, y, y_pred, nodes
    for dataset in dataset_list:
        pkl.dump(results[dataset], open(f'results/results_wta_s-{s}_{dataset}.pkl', 'wb'))


def test_wta_large(num_cpus=1, num_trials=20, s=5):
    dataset_list = ['flickr', 'youtube', 'ogbn-products', 'ogbn-arxiv']
    weights = [False, False, False, False]
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        for trial_i in range(num_trials):
            np.random.seed(trial_i)
            nodes = np.random.permutation(n)
            para = (dataset, trial_i, csr_mat, y, s, trial_i, nodes)
            para_space.append(para)
    pool = multiprocessing.Pool(processes=num_cpus)
    pool_results = pool.map(func=run_weighted_tree_algorithm, iterable=para_space)
    pool.close()
    pool.join()
    results = dict()
    for dataset in dataset_list:
        results[dataset] = dict()
    for dataset, trial_i, rt, y, y_pred, nodes in pool_results:
        results[dataset][(trial_i, 'wta')] = rt, y, y_pred, nodes
    for dataset in dataset_list:
        pkl.dump(results[dataset], open(f'results/results_wta_s-{s}_{dataset}.pkl', 'wb'))


def main():
    args = sys.argv[1:]
    if args[3] == 'small':
        test_wta_small(num_cpus=int(args[0]), num_trials=int(args[1]), s=int(args[2]))
    elif args[3] == 'large':
        test_wta_large(num_cpus=int(args[0]), num_trials=int(args[1]), s=int(args[2]))
    else:
        test_weighted_tree_simple()


main()

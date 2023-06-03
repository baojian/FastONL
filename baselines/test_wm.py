import os
import sys
import numpy as np
from algo import algo_weighted_majority
from utils import load_graph_data
import multiprocessing
import pickle as pkl

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def run_weighted_majority(para):
    dataset, trial_i, csr_mat, y, nodes, seed = para
    y_pred, rt1 = algo_weighted_majority(csr_mat, y, nodes, seed)
    return dataset, trial_i, rt1, y, y_pred, nodes


def test_weighted_majority_small(num_cpus=1, num_trials=20):
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10', 'blogcatalog']
    weights = [False, False, False, False, True, False, False]
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        print(dataset, n, len(y), np.sum(csr_mat.data) / 2, len(set(y)))
        for trial_i in range(num_trials):
            np.random.seed(trial_i)
            nodes = np.random.permutation(n)
            para_space.append((dataset, trial_i, csr_mat, y, nodes, trial_i))
    pool = multiprocessing.Pool(processes=num_cpus)
    pool_results = pool.map(func=run_weighted_majority, iterable=para_space)
    pool.close()
    pool.join()
    results = dict()
    for dataset in dataset_list:
        results[dataset] = dict()
    for dataset, trial_i, rt1, y, y_pred1, nodes in pool_results:
        results[dataset][(trial_i, 'weighted-majority')] = rt1, y, y_pred1, nodes
    for dataset in dataset_list:
        pkl.dump(results[dataset], open(f'results/results_weighted-majority_{dataset}.pkl', 'wb'))


def test_weighted_majority_large(num_cpus=1, num_trials=20):
    dataset_list = ['flickr', 'youtube', 'ogbn-products', 'ogbn-arxiv']
    weights = [False, False, False, False]
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        print(dataset, n, len(y), np.sum(csr_mat.data) / 2, len(set(y)))
        for trial_i in range(num_trials):
            np.random.seed(trial_i)
            nodes = np.random.permutation(n)
            para_space.append((dataset, trial_i, csr_mat, y, nodes, trial_i))
    pool = multiprocessing.Pool(processes=num_cpus)
    pool_results = pool.map(func=run_weighted_majority, iterable=para_space)
    pool.close()
    pool.join()
    results = dict()
    for dataset in dataset_list:
        results[dataset] = dict()
    for dataset, trial_i, rt1, y, y_pred1, nodes in pool_results:
        results[dataset][(trial_i, 'weighted-majority')] = rt1, y, y_pred1, nodes
    for dataset in dataset_list:
        pkl.dump(results[dataset], open(f'results/results_weighted-majority_{dataset}.pkl', 'wb'))


def main():
    args = sys.argv[1:]
    if args[2] == 'small':
        test_weighted_majority_small(num_cpus=int(args[0]), num_trials=int(args[1]))
    elif args[2] == 'large':
        test_weighted_majority_large(num_cpus=int(args[0]), num_trials=int(args[1]))


if __name__ == '__main__':
    main()

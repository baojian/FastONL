import os
import sys
import numpy as np
import multiprocessing
import pickle as pkl
from utils import load_graph_data
from algo import algo_online_rakhlin

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def run_online_rakhlin(para):
    dataset, trial_i, para_ind, csr_mat, kappa, y, nodes, mat_type = para
    y_hat1, rt1 = algo_online_rakhlin(csr_mat, kappa, y, nodes, mat_type)
    return dataset, trial_i, para_ind, kappa, rt1, y, y_hat1, nodes


def test_relaxation_small(num_cpus=1, num_trials=20):
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10', 'blogcatalog']
    weights = [False, False, False, False, True, False, False]
    mat_type = 'normalized'
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        kappa_list = np.arange(0.05, 1.01, 0.05) * n
        for trial_i in range(num_trials):
            np.random.seed(trial_i)
            nodes = np.random.permutation(n)
            for para_ind, kappa in enumerate(kappa_list):
                para = (dataset, trial_i, para_ind, csr_mat, kappa, y, nodes, mat_type)
                para_space.append(para)
    pool = multiprocessing.Pool(processes=num_cpus)
    pool_results = pool.map(func=run_online_rakhlin, iterable=para_space)
    pool.close()
    pool.join()
    results = dict()
    for dataset in dataset_list:
        results[dataset] = dict()
    for dataset, trial_i, para_ind, kappa, rt, y, y_pred, nodes in pool_results:
        results[dataset][(trial_i, para_ind, 'relaxation')] = rt, y, y_pred, nodes
    for dataset in dataset_list:
        pkl.dump(results[dataset], open(f'results/results_relaxation-{mat_type}_{dataset}.pkl', 'wb'))


def test_relaxation_large(num_cpus=1, num_trials=20):
    dataset_list = ['flickr', 'youtube', 'ogbn-products', 'ogbn-arxiv']
    weights = [False, False, False, False]
    mat_type = 'normalized'
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        kappa_list = np.arange(0.05, 1.01, 0.05) * n
        for trial_i in range(num_trials):
            np.random.seed(trial_i)
            nodes = np.random.permutation(n)
            for para_ind, kappa in enumerate(kappa_list):
                para = (dataset, trial_i, para_ind, csr_mat, kappa, y, nodes, mat_type)
                para_space.append(para)
    pool = multiprocessing.Pool(processes=num_cpus)
    pool_results = pool.map(func=run_online_rakhlin, iterable=para_space)
    pool.close()
    pool.join()
    results = dict()
    for dataset in dataset_list:
        results[dataset] = dict()
    for dataset, trial_i, para_ind, kappa, rt, y, y_pred, nodes in pool_results:
        results[dataset][(trial_i, para_ind, 'relaxation')] = rt, y, y_pred, nodes
    for dataset in dataset_list:
        pkl.dump(results[dataset], open(f'results/results_relaxation-{mat_type}_{dataset}.pkl', 'wb'))


def main():
    args = sys.argv[1:]
    if args[2] == 'small':
        test_relaxation_small(num_cpus=int(args[0]), num_trials=int(args[1]))
    elif args[2] == 'large':
        test_relaxation_large(num_cpus=int(args[0]), num_trials=int(args[1]))


main()

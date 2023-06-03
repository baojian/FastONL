import os
import sys
import numpy as np
from algo import algo_inverse_approx
from utils import load_graph_data
import multiprocessing
import pickle as pkl

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def run_inverse_approx(para):
    dataset, trial_i, para_ind, csr_mat, kappa, y, nodes, num_iter = para
    y_pred, rt, nnz = algo_inverse_approx(csr_mat, kappa, y, nodes, num_iter)
    return dataset, trial_i, para_ind, kappa, rt, nnz, y, y_pred, nodes


def test_inverse_approx_small(num_cpus=1, num_trials=20, num_iter=1):
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10', 'blogcatalog']
    weights = [False, False, False, False, True, False, False]
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        kappa_list = np.arange(0.05, 1.01, 0.05) * n
        for trial_i in range(num_trials):
            np.random.seed(trial_i)
            nodes = np.random.permutation(n)
            for para_ind, kappa in enumerate(kappa_list):
                para_space.append((dataset, trial_i, para_ind, csr_mat, kappa, y, nodes, num_iter))
    pool = multiprocessing.Pool(processes=num_cpus)
    pool_results = pool.map(func=run_inverse_approx, iterable=para_space)
    pool.close()
    pool.join()
    results = dict()
    for dataset in dataset_list:
        results[dataset] = dict()
    for dataset, trial_i, para_ind, kappa, rt, nnz, y, y_pred, nodes in pool_results:
        results[dataset][(trial_i, para_ind, 'inverse-approx')] = rt, nnz, y, y_pred, nodes
    for dataset in dataset_list:
        pkl.dump(results[dataset], open(
            f'results/results_inverse-approx_num-iter-{num_iter}_{dataset}.pkl', 'wb'))


def test_inverse_approx_large(num_cpus=1, num_trials=20, num_iter=1):
    dataset_list = ['flickr', 'youtube', 'ogbn-products', 'ogbn-arxiv']
    weights = [False, False, False, False]
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        kappa_list = np.arange(0.05, 1.01, 0.05) * n
        for trial_i in range(num_trials):
            np.random.seed(trial_i)
            nodes = np.random.permutation(n)
            for para_ind, kappa in enumerate(kappa_list):
                para_space.append((dataset, trial_i, para_ind, csr_mat, kappa, y, nodes, num_iter))
    pool = multiprocessing.Pool(processes=num_cpus)
    pool_results = pool.map(func=run_inverse_approx, iterable=para_space)
    pool.close()
    pool.join()
    results = dict()
    for dataset in dataset_list:
        results[dataset] = dict()
    for dataset, trial_i, para_ind, kappa, rt, nnz, y, y_pred, nodes in pool_results:
        results[dataset][(trial_i, para_ind, 'inverse-approx')] = rt, nnz, y, y_pred, nodes
    for dataset in dataset_list:
        pkl.dump(results[dataset], open(
            f'results/results_inverse-approx_num-iter-{num_iter}_{dataset}.pkl', 'wb'))


def main():
    args = sys.argv[1:]
    if args[3] == 'small':
        test_inverse_approx_small(num_cpus=int(args[0]), num_trials=int(args[1]), num_iter=int(args[2]))
    elif args[3] == 'large':
        test_inverse_approx_large(num_cpus=int(args[0]), num_trials=int(args[1]), num_iter=int(args[2]))


if __name__ == '__main__':
    main()

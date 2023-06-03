import argparse
import numpy as np
import scipy.sparse as sp
from algo import fast_onc


def datasets():
    dataset_list = ['political-blog', 'citeseer', 'cora', 'mnist-tr-nei10',
                    'pubmed', 'blogcatalog', 'youtube', 'ogbn-arxiv']
    weighted_dict = {'political-blog': False, 'citeseer': False, 'cora': False,
                     'pubmed': False, 'mnist-tr-nei10': True, 'blogcatalog': False,
                     'youtube': False, 'ogbn-arxiv': False}
    return dataset_list, weighted_dict


def load_graph_data(dataset='citeseer'):
    if dataset not in datasets()[0]:
        print(f"{dataset} does not exist!")
        exit(0)
    f = np.load(f'./datasets/{dataset}.npz')
    csr_mat = sp.csr_matrix(
        (f["data"], f["indices"], f["indptr"]), shape=f["shape"])
    labels = f['labels']
    return csr_mat, labels


def main(args):
    np.random.seed(args.seed)
    dataset_list, weighted_dict = datasets()
    if args.dataset not in dataset_list:
        print(f'Please choose dataset from {dataset_list}')
        exit(0)
    csr_mat, y = load_graph_data(dataset=args.dataset)
    n, _ = csr_mat.shape
    nodes = np.random.permutation(n)
    kernel_list = ['k1', 'k2', 'k3', 'k4']
    if args.kernel not in kernel_list:
        print(f'Please choose kernel from {kernel_list}')
        exit(0)
    eps = args.eps
    weighted = weighted_dict[args.dataset]
    lambda_ = args.lambda_
    if args.kernel == 'k1':
        kernel_id = 1
    elif args.kernel == 'k2':
        kernel_id = 2
    elif args.kernel == 'k3':
        kernel_id = 3
    else:
        kernel_id = 4
    acc, y_hat, rt = fast_onc(
        csr_mat, weighted, kernel_id, eps, lambda_, y, nodes)
    print(f"run on {args.dataset} dataset with {n} nodes, using kernel {args.kernel}\n"
          f"accuracy: {acc:.3f} run time: {rt:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastONL')
    # Credits to GPT-4
    # String argument for kernel
    parser.add_argument('--kernel', type=str, default='k1',
                        required=False, help='Kernel type')
    # String argument for dataset
    parser.add_argument('--dataset', type=str, default='citeseer',
                        required=False, help='Dataset name')
    # Float argument for lambda
    parser.add_argument('--lambda_', type=float, default=100,
                        required=False, help='Alpha value')
    # Float argument for eps
    parser.add_argument('--eps', type=float, default=1e-5,
                        required=False, help='Epsilon value')
    parser.add_argument('--seed', type=int, default=17,
                        required=False, help='Epsilon value')
    main(parser.parse_args())

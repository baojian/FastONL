import argparse
import numpy as np
from algo import fast_onc
from utils import load_graph_data


def main(args):
    dataset = args.dataset
    weighted = False
    eps = 1e-5
    np.random.seed(0)
    csr_mat, y = load_graph_data(dataset=dataset)
    n, _ = csr_mat.shape
    nodes = np.random.permutation(n)
    for kernel_id in range(1, 5):
        print('-' * 20 + f'kernel-{kernel_id}' + '-' * 20)
        for lambda_ in np.arange(0.1, 1., 0.1) * n:
            acc, y_hat, rt = fast_onc(
                csr_mat, weighted, kernel_id, eps, lambda_, y, nodes)
            print(f'lambda: {lambda_:.1f} , acc: {acc:.5f}, run-time: {rt:.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of FastONL')
    # String argument for dataset
    parser.add_argument('--dataset', type=str, default='citeseer',
                        required=False, help='Dataset name')
    main(parser.parse_args())

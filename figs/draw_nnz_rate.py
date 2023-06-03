import time
import multiprocessing
import pickle as pkl
import numpy as np
import seaborn as sns
import scipy.sparse as sp
from utils import load_graph_data
from matplotlib import pyplot as plt


def run_inverse_approx(para):
    csr_mat, dataset, kappa, y, nodes, num_iter = para
    n, _ = csr_mat.shape
    assert len(nodes) == len(y)
    diags = csr_mat.sum(axis=1).A.flatten()
    L = (n / (n + kappa)) * csr_mat
    with np.errstate(divide="ignore"):
        diags_sqrt = 1. / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], n, n, format="csr")
    KK = DH @ (L @ DH)
    YY = KK
    start_time = time.time()
    ZZ = sp.eye(n)
    nnz_rate = [n / n ** 2.]
    run_time_list = [time.time() - start_time]
    for i in range(num_iter):
        start_time = time.time()
        ZZ += YY
        YY = YY @ KK
        nnz_rate.append(ZZ.nnz / n ** 2.)
        print(dataset, i, ZZ.nnz / n ** 2.)
        run_time_list.append(time.time() - start_time)
    return dataset, nnz_rate, run_time_list


def run_nnz_rate():
    num_iter = 20
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10' 'blogcatalog']
    weights = [False, False, False, False, True, False, False]
    para_space = []
    for dataset, weighted in zip(dataset_list, weights):
        csr_mat, y = load_graph_data(dataset=dataset)
        n, _ = csr_mat.shape
        kappa = .15 * n
        nodes = range(n)
        para_space.append((csr_mat, dataset, kappa, y, nodes, num_iter))
    pool = multiprocessing.Pool(processes=10)
    pool_results = pool.map(func=run_inverse_approx, iterable=para_space)
    pkl.dump(pool_results, open('baselines/fig-inverse-approx-nnz-rate.pkl', 'wb'))
    pool.close()
    pool.join()


def draw_nnz_rate():
    plt.rc('font', family='serif', serif='Times New Roman')
    font = {'family': "Times New Roman",
            'weight': 'bold',
            'size': 20}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    sns.set()
    results = pkl.load(open('baselines/fig-inverse-approx-nnz-rate.pkl', 'rb'))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    dataset_title_list = ['Political', 'Citeseer', 'Cora', 'Pudmed', 'MNIST', 'Blogcatalog']
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed', 'mnist-tr-nei10', 'blogcatalog']
    for dataset, nnz_rate, run_time_list in results:
        x = range(len(nnz_rate))
        ind = dataset_list.index(dataset)
        ax[0].plot(x, nnz_rate, marker='D', label=dataset_title_list[ind])
        ax[1].plot(x, np.cumsum(run_time_list), marker='s', label=dataset_title_list[ind])
    for i in range(2):
        ax[i].grid(b=True, which='major')
        ax[i].grid(b=True, which='minor')
    ax[0].legend()
    ax[0].set_xticks([5, 10, 15, 20])
    ax[1].set_xticks([5, 10, 15, 20])
    ax[0].set_xticklabels([5, 10, 15, 20])
    ax[1].set_xticklabels([5, 10, 15, 20])
    ax[0].set_xlabel(r'$p$')
    ax[1].set_xlabel(r'$p$')
    ax[0].set_ylabel(r'$\operatorname{nnz}$ rate', fontsize=20)
    ax[1].set_ylabel(r'Run time (seconds)', fontsize=20)
    plt.subplots_adjust(wspace=0.22, hspace=0.1)
    fig.savefig(f"figs/fig-nnz-rate.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


if __name__ == '__main__':
    draw_nnz_rate()

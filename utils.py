import time
import numpy as np
from numba import njit
import scipy.sparse as sp
from scipy.sparse import csr_matrix


@njit(cache=True)
def loss_grad_val(psi_t, y, st, loss_type):
    y = int(y)
    if loss_type == 'hinge':
        if y in st:
            loss = 1. - psi_t[y] + (np.sum(psi_t[st]) - 1.) / len(st)
            grad = np.zeros(len(psi_t))
            grad[st] = 1. / len(st)
            grad[y] -= 1.
            return loss, grad
        else:
            max_val, max_ind = - np.infty, 0
            for i in range(len(psi_t)):
                if y == i:
                    continue
                val = psi_t[i] - psi_t[y]
                if val > max_val:
                    max_val = val
                    max_ind = i
            factor = 1. / (1. + 1. / len(st))
            loss = (1. + max_val) * factor
            grad = np.zeros(len(psi_t))
            grad[max_ind] = factor
            grad[y] = -factor
            return loss, grad


def load_graph_data(dataset='citeseer'):
    if dataset in ['political-blog', 'citeseer', 'cora',
                   'mnist-tr-nei10', 'pubmed', 'blogcatalog']:
        f = np.load(f'./datasets/{dataset}.npz')
        csr_mat = csr_matrix((f["data"], f["indices"], f["indptr"]), shape=f["shape"])
        labels = f['labels']
        return csr_mat, labels
    elif dataset in ['flickr', 'youtube', 'ogbn-products', 'ogbn-arxiv']:
        f = np.load(f'./dataset/{dataset}.npz')
        csr_mat = csr_matrix((f["data"], f["indices"], f["indptr"]), shape=f["shape"])
        labels = f['labels']
        return csr_mat, labels


def laplacian_mat(adj_mat, lap_type):
    # to get I - D^{-1/2} A D^{-1/2}
    if lap_type == 'normalized':
        n, m = adj_mat.shape
        diags = adj_mat.sum(axis=1).A.flatten()
        D = sp.spdiags(diags, [0], m, n, format="csr")
        L = D - adj_mat
        with np.errstate(divide="ignore"):
            diags_sqrt = 1. / np.sqrt(diags)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        DH = sp.spdiags(diags_sqrt, [0], m, n, format="csr")
        return DH @ (L @ DH)
    # to get D - A
    elif lap_type == 'un_normalized':
        A = adj_mat
        n, m = A.shape
        diags = A.sum(axis=1).A.flatten()
        D = sp.spdiags(diags, [0], m, n, format="csr")
        L = D - A
        return L
    else:
        return None


@njit(cache=True)
def find_water_filling(psi):
    """
    Given an array of scores \\psi_t, find the distribution of
    this array based on water-filling method. Current time
    complexity is O(k \\log k).
    :param psi: a list of scores.
    :return:
    """
    # get indices of dist_psi in decreasing order.
    sorted_indices = np.argsort(psi)
    k = len(psi)
    sum_psi = np.sum(psi)
    for ii in range(len(psi)):
        tau = (sum_psi - 1.) / k
        if (psi[sorted_indices[ii]] - tau) >= 0:
            qt = [max(_, 0) for _ in psi - tau]
            st = sorted_indices[ii:]
            return qt, st
        k -= 1
        sum_psi -= psi[sorted_indices[ii]]
    # never reach this point.
    return None


def find_wilson_rst(n, indptr, indices, data, root):
    in_tree = np.zeros(n, dtype=np.bool_)
    next_ = np.zeros(n, dtype=np.int32)
    weights_ = np.zeros(n, dtype=float)
    next_[root] = root
    in_tree[root] = True
    tree_edges = []
    for i in range(n):
        u = i
        while not in_tree[u]:
            nei = indptr[u + 1] - indptr[u]
            nodes = np.zeros(nei, dtype=np.int32)
            weights = np.zeros(nei, dtype=float)
            total_wei = 0.
            for jj, v_ind in enumerate(range(indptr[u], indptr[u + 1])):
                nodes[jj] = indices[v_ind]
                weights[jj] = data[v_ind]
                total_wei += data[v_ind]
            p = weights / total_wei
            ind = np.random.choice(a=range(len(nodes)), replace=True, p=p)
            next_[u] = nodes[ind]
            weights_[u] = weights[ind]
            u = next_[u]
        u = i
        while not in_tree[u]:
            in_tree[u] = True
            tree_edges.append((u, next_[u], weights_[u]))
            u = next_[u]
    return next_, tree_edges


def test_find_water_fill():
    """
    Test water-filling method.
    :return:
    """
    print(np.random.normal(0., 0.2, 10))
    np.random.seed(4)
    psi0 = [-0.0, -0.0]
    qt, st = find_water_filling(np.asarray(psi0))
    print(qt, st, np.sum(qt))
    psi1 = [0.05056171, 0.49995133, -0.99590893, 0.69359851, -0.41830152]
    qt, st = find_water_filling(np.asarray(psi1))
    print(qt, st, np.sum(qt))
    psi2 = [0.15, 0.15, 0.15, 0.15, 0.15]
    qt, st = find_water_filling(np.asarray(psi2))
    print(qt, st, np.sum(qt))
    psi3 = [-1., -1., 1.]
    qt, st = find_water_filling(np.asarray(psi3))
    print(qt, st, np.sum(qt))
    psi4 = [-1.]
    qt, st = find_water_filling(np.asarray(psi4))
    print(qt, st, np.sum(qt))
    psi5 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    qt, st = find_water_filling(np.asarray(psi5))
    print(qt, st, np.sum(qt))
    psi6 = [-4.98479718e-02, 7.94462166e-03, -1.47659985e-01, -1.31876652e-01,
            -1.57832827e-01, -3.18498562e-01, 2.56191247e-01, -1.51318288e-04,
            7.57976543e-02, 3.00235140e-01]
    qt, st = find_water_filling(np.asarray(psi6))
    print(qt, st, np.sum(qt))


def parse_results_small():
    import pickle as pkl
    import matplotlib.pyplot as plt
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10', 'blogcatalog']
    method_list = ['fast-onc', 'weighted-majority']
    for dataset in dataset_list:
        acc_mat = dict()
        rt_mat = dict()
        for method in method_list:
            if method == 'fast-onc':
                eps = '5e-4'
                results = pkl.load(open(f'baselines/results_{method}_eps-{eps}_{dataset}.pkl', 'rb'))
                for (trial_i, para_ind, method_v), (rt, y, y_pred, nodes) in results.items():
                    if method_v not in rt_mat:
                        rt_mat[method_v] = np.zeros(shape=(20, 20))
                    if method_v not in acc_mat:
                        acc_mat[method_v] = np.zeros(shape=(20, 20))
                    rt_mat[method_v][trial_i][para_ind] = rt
                    match = [y_pred[_] == y[_] for _ in nodes if y[_] != -1]
                    acc = np.sum(match) / len(match)
                    acc_mat[method_v][trial_i][para_ind] = acc
            if method == 'weighted-majority':
                results = pkl.load(open(f'baselines/results_{method}_{dataset}.pkl', 'rb'))
                for (trial_i, method_v), (rt, y, y_pred, nodes) in results.items():
                    if method_v not in rt_mat:
                        rt_mat[method_v] = np.zeros(shape=20)
                    if method_v not in acc_mat:
                        acc_mat[method_v] = np.zeros(shape=20)
                    rt_mat[method_v][trial_i] = rt
                    match = [y_pred[_] == y[_] for _ in nodes if y[_] != -1]
                    acc = np.sum(match) / len(match)
                    acc_mat[method_v][trial_i] = acc
        fig, ax = plt.subplots(1, 1)
        for method in acc_mat:
            print(dataset, method, np.mean(acc_mat[method], axis=0))
            xx = np.mean(acc_mat[method], axis=0)
            if type(xx) == np.float64:
                ax.plot([xx] * 20, label=method)
            else:
                ax.plot(xx, label=method)
        plt.legend()
        plt.show()


def test_forward_push(
        weighted=False, dataset='citeseer', eps=1e-8):
    from algo import GraphKernel
    csr_mat, labels = load_graph_data(dataset=dataset)
    if weighted:
        np.random.seed(17)
        data = np.random.uniform(1., 2., len(csr_mat.data))
        csr_mat.data = data
    n, _ = csr_mat.shape
    kappa = 0.15 * n
    alpha = 1. - n / (n + kappa)
    gk = GraphKernel(csr_mat, alpha, eps, weighted=False)
    total_time = time.time()
    deg = csr_mat.sum(1).A.flatten()
    sqrt_deg = np.sqrt(deg)
    for s in range(5):
        v1 = gk.forward_push_m2(s)
        ppr_vec = (v1 / sqrt_deg) * sqrt_deg[s]
        print(np.linalg.norm(ppr_vec, 1))
    total_time = time.time() - total_time
    print(f'total run time is {total_time:.2f} seconds')


def test_forward_push_m1(
        dataset='citeseer', eps=1e-9, alpha=0.15):
    from algo import GraphKernel
    csr_mat, labels = load_graph_data(dataset=dataset)
    csr_mat.data = np.asarray(csr_mat.data, dtype=np.float64)
    n, _ = csr_mat.shape
    lap = laplacian_mat(adj_mat=csr_mat, lap_type='un_normalized')
    m1 = (lap + alpha * sp.eye(n)).todense()
    m2 = np.linalg.inv(m1)
    gk = GraphKernel(csr_mat, alpha, eps, weighted=False)
    for s in range(5):
        start_time = time.time()
        vec = gk.forward_push_m1(s)
        rt = time.time() - start_time
        print(f'{s} l1-error:{np.sum(np.abs(vec - m2[s])):6e} rt:{rt:4e}')


def test_algo_rakhlin(dataset='citeseer'):
    import networkx as nx
    from algo import algo_online_rakhlin
    num_trials = 1
    np.random.seed(17)
    csr_mat, labels = load_graph_data(dataset=dataset)
    g = nx.from_scipy_sparse_array(csr_mat)
    print(f"dataset:{dataset} num-cc:{nx.number_connected_components(g)}")
    n, _ = csr_mat.shape
    y = labels
    kappa_list = np.arange(0.1, 1.01, 0.05) * n
    alpha_list = np.arange(0.05, 1.01, 0.05)
    for trial_i in range(num_trials):
        for ind, (kappa, alpha) in enumerate(zip(kappa_list, alpha_list)):
            nodes = np.random.permutation(n)
            acc, run_time = algo_online_rakhlin(csr_mat, kappa, y, nodes)
            print(trial_i, ind, acc, run_time)


def test_weighted_tree():
    from algo import algo_weighted_tree
    num_trials = 1
    np.random.seed(17)
    dataset = 'citeseer'
    csr_mat, labels = load_graph_data(dataset=dataset)
    n, _ = csr_mat.shape
    y = labels
    s = 5  # for 5 random spanning trees
    for trial_i in range(num_trials):
        nodes = np.random.permutation(n)
        y_pred, rt = algo_weighted_tree(csr_mat, y, nodes, s, 17, verbose=False)
        match = [y_pred[_] == y[_] for _ in nodes if y[_] != -1]
        acc = np.sum(match) / len(match)
        print(trial_i, acc, rt)


def test_fast_onc(dataset='citeseer'):
    from algo import algo_fast_onc_k1
    num_trials = 1
    np.random.seed(17)
    csr_mat, labels = load_graph_data(dataset=dataset)
    n, _ = csr_mat.shape
    y = labels
    weighted = False
    eps = 1e-5
    lambda_list = np.arange(0.05, 1.01, 0.05) * n
    for trial_i in range(num_trials):
        for ind, (lambda_) in enumerate(lambda_list):
            nodes = np.random.permutation(n)
            acc, rt = algo_fast_onc_k1(csr_mat, weighted, lambda_, y, nodes, eps)
            print(trial_i, ind, acc, rt)

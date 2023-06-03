import time
import numpy as np
from numba import njit
import scipy.sparse as sp
from utils import loss_grad_val
from utils import laplacian_mat
from utils import find_water_filling
from utils import find_wilson_rst


@njit(cache=True)
def _forward_push_m1(
        indptr, indices, data, weighted, degree, alpha, eps, s):
    """
    Given any node s, to calculate the s-th column of
    (alpha*I + D - A)^{-1} by using generalized APPR.

    :param s: the source node s
    :return:
    """
    n = len(degree)
    queue = np.zeros(n, dtype=np.int32)
    front = np.int32(0)
    rear = np.int32(1)
    re_vec = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    queue[rear] = s
    r[s] = 1. / alpha
    q_mark = np.zeros(n, dtype=np.bool_)
    q_mark[s] = True
    eps_vec = np.zeros(n, dtype=np.float64)
    for i in np.arange(n):
        eps_vec[i] = degree[i] * eps
    if weighted:
        while rear != front:
            front = (front + 1) % n
            u = queue[front]
            q_mark[u] = False
            if r[u] > eps_vec[u]:
                res_u = r[u]
                re_vec[u] += alpha * res_u / (alpha + degree[u])
                r[u] = 0
                push_amount = res_u / (alpha + degree[u])
                for v_ind in range(indptr[u], indptr[u + 1]):
                    v = indices[v_ind]
                    wei = data[v_ind]
                    r[v] += push_amount * wei
                    if not q_mark[v]:
                        rear = (rear + 1) % n
                        queue[rear] = v
                        q_mark[v] = True
        return re_vec
    else:
        while rear != front:
            front = (front + 1) % n
            u = queue[front]
            q_mark[u] = False
            if r[u] > eps_vec[u]:
                res_u = r[u]
                re_vec[u] += alpha * res_u / (alpha + degree[u])
                r[u] = 0
                push_amount = res_u / (alpha + degree[u])
                for v in indices[indptr[u]:indptr[u + 1]]:
                    r[v] += push_amount
                    if not q_mark[v]:
                        rear = (rear + 1) % n
                        queue[rear] = v
                        q_mark[v] = True
        return re_vec


@njit(cache=True)
def _forward_push_m2(indptr, indices, data, weighted, degree, alpha, eps, s):
    """
    Given any node s, to calculate the s-th column of
    alpha*(I -(1-alpha)WD^{-1})^{-1} by using the APPR algorithm.

    Given any node s, we return its PPR vector by using
    Forward-Push algorithm (a.k.a. coordinate descent).
    :param s: the source node s
    :return:
    """
    n = len(degree)
    queue = np.zeros(n, dtype=np.int32)
    front = np.int32(0)
    rear = np.int32(1)
    re_vec = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    queue[rear] = s
    r[s] = 1.
    q_mark = np.zeros(n, dtype=np.bool_)
    q_mark[s] = True
    eps_vec = np.zeros(n, dtype=np.float64)
    for i in np.arange(n):
        eps_vec[i] = degree[i] * eps
    if weighted:
        while rear != front:
            front = (front + 1) % n
            u = queue[front]
            q_mark[u] = False
            if r[u] > eps_vec[u]:
                res_u = r[u]
                re_vec[u] += alpha * res_u
                rest_prob = (1. - alpha) / degree[u]
                r[u] = 0
                push_amount = rest_prob * res_u
                for v_ind in range(indptr[u], indptr[u + 1]):
                    v = indices[v_ind]
                    wei = data[v_ind]
                    r[v] += push_amount * wei
                    if not q_mark[v]:
                        rear = (rear + 1) % n
                        queue[rear] = v
                        q_mark[v] = True
        return re_vec
    else:
        while rear != front:
            front = (front + 1) % n
            u = queue[front]
            q_mark[u] = False
            if r[u] > eps_vec[u]:
                res_u = r[u]
                re_vec[u] += alpha * res_u
                rest_prob = (1. - alpha) / degree[u]
                r[u] = 0
                push_amount = rest_prob * res_u
                for v in indices[indptr[u]:indptr[u + 1]]:
                    r[v] += push_amount
                    if not q_mark[v]:
                        rear = (rear + 1) % n
                        queue[rear] = v
                        q_mark[v] = True
        return re_vec


@njit(cache=True)
def _forward_push_bounds(indptr, indices, degree, alpha, eps, s):
    """
    Given any node s, we return its PPR vector by using
    Forward-Push algorithm (a.k.a. coordinate descent).
    :param s: the source node s
    :return:
    """
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int32)
    front = np.int32(0)
    rear = np.int32(2)
    re_vec = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    queue[1] = s
    queue[rear] = -1  # ddag in our paper
    r[s] = 1.
    q_mark = np.zeros(n + 1, dtype=np.bool_)
    q_mark[s] = True
    q_mark[n] = True
    eps_vec = np.zeros(n, dtype=np.float64)
    t = 0
    for i in np.arange(n):
        eps_vec[i] = degree[i] * eps

    vol_active = 0
    vol_total = degree[s]
    aver_active_vol = 0.
    ratio_list = []

    num_operations = 0.
    sum_r = 1.
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == -1:
            # the last epoch may not have any active nodes
            if vol_active <= 0.:
                rear = (rear + 1) % n
                queue[rear] = -1
                q_mark[n] = True
                continue
            aver_active_vol += vol_active
            ratio_list.append(vol_active / vol_total)
            vol_active = 0.
            vol_total = 0.
            for _ in range(n):
                if r[_] != 0.:
                    vol_total += degree[_]
            t = t + 1
            rear = (rear + 1) % n
            queue[rear] = -1
            q_mark[n] = True
            continue
        if r[u] >= eps_vec[u]:
            num_operations += degree[u]

            vol_active += degree[u]
            res_u = r[u]
            re_vec[u] += alpha * res_u
            sum_r -= alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True

    aver_ratio = 0.
    for _ in ratio_list:
        aver_ratio += _
    if t > 0:
        aver_ratio /= t
        aver_active_vol /= t
    st = 0
    for i in range(n):
        if r[i] != 0.:
            st += 1
    theoretical_bound = 0.
    estimate_t = 0
    if aver_ratio > 0.:
        theoretical_bound = (aver_active_vol / (alpha * aver_ratio)) * np.log(1. / (eps * (1. - alpha) * st))
        estimate_t = (1. / (alpha * aver_ratio)) * np.log(1. / (eps * (1. - alpha) * st))
    if num_operations > theoretical_bound:
        print(r"error", num_operations, theoretical_bound)
    return sum_r, aver_active_vol, aver_ratio, t, estimate_t, num_operations, theoretical_bound, re_vec


# Approximate Graph Kernel Operator
class GraphKernel:

    def __init__(self, csr_mat, alpha, eps, weighted):
        self.csr_mat = csr_mat
        self.indices = csr_mat.indices
        self.indptr = csr_mat.indptr
        self.data = csr_mat.data
        self.degree = csr_mat.sum(axis=1).A.flatten()

        self.n = csr_mat.shape[0]
        self.eps = eps
        self.alpha = alpha
        self.weighted = weighted

    def forward_push_m1(self, s):
        """
        To calculate the s-th column of (alpha*I + D - W)^{-1}
        :param s source node s
        I:
        """
        return _forward_push_m1(self.indptr, self.indices, self.data, self.weighted,
                                self.degree, self.alpha, self.eps, s)

    def forward_push_m2(self, s):
        """ Given any node s, we return the PPR vector. """
        return _forward_push_m2(self.indptr, self.indices, self.data, self.weighted,
                                self.degree, self.alpha, self.eps, s)

    def forward_push_bounds(self, s):
        """ Given any node s, we return the PPR vector. """
        return _forward_push_bounds(self.indptr, self.indices,
                                    self.degree, self.alpha, self.eps, s)


def fast_onc(csr_mat, weighted, kernel_id, eps, lambda_, y, nodes):
    """
    FastONL proposed in [1].
    ---
    [1] Zhou, Baojian, Yifan Sun, and Reza Babanezhad.
        "Fast Online Node Labeling for Very Large Graphs."
        arXiv preprint arXiv:2305.16257 (2023).
    """

    start_time = time.time()
    n, _ = csr_mat.shape
    labels = [int(_) for _ in set(y)]
    k = len(labels) if -1 not in labels else len(labels) - 1
    assert len(nodes) == len(y)
    assert kernel_id in [1, 2, 3, 4]
    # kernel id is listed in Table 1 of [1].
    if kernel_id == 1:
        alpha = lambda_ / n
        coeff = 2. * lambda_
    elif kernel_id == 2:
        alpha = lambda_ / (n + lambda_)
        coeff = 2. * n
    elif kernel_id == 3:
        alpha = 1. - lambda_ / (n + lambda_)
        coeff = 2. * lambda_
    else:
        beta = 1. - lambda_ / n
        # alpha is actually 1.
        alpha = (n * beta + lambda_) / n
        coeff = 2. * lambda_
    gk = GraphKernel(csr_mat, alpha, eps, weighted)
    deg = csr_mat.sum(1).A.flatten()
    sqrt_deg = np.sqrt(deg)

    a = 0.
    g = np.zeros((k, n))
    y_hat = np.zeros(len(nodes))
    # You can use another big constant such as t = k * (n ** 2.)
    t = 2. * (n ** 2.)

    for ii, uu in enumerate(nodes):
        if int(y[uu]) == -1:  # ignore the unlabeled nodes
            continue
        if kernel_id == 1:
            v1 = gk.forward_push_m1(uu)
            re_vec = coeff * v1
        elif kernel_id == 2:
            v2 = gk.forward_push_m2(uu)
            re_vec = coeff * ((v2 / sqrt_deg) * sqrt_deg[uu])
        elif kernel_id == 3:
            v3 = gk.forward_push_m2(uu)
            re_vec = coeff * ((v3 / sqrt_deg) * sqrt_deg[uu])
        else:
            # We set S^{-1/2} as D^{-1/2}
            # This may give better performance than we reported in Fig. 6.
            v4 = gk.forward_push_m1(uu)
            re_vec = coeff * ((v4 / sqrt_deg) * sqrt_deg[uu])
        v = np.dot(g, re_vec)
        psi_t = -v / np.sqrt(a + k * t)
        qt, st = find_water_filling(psi_t)
        y_hat[uu] = int(np.argmax(qt))
        loss, grad = loss_grad_val(psi_t, int(y[uu]), st, loss_type='hinge')
        g[:, uu] = grad
        a += (2. * np.dot(grad, v) + re_vec[uu] * np.linalg.norm(grad) ** 2.)
        t -= re_vec[uu]
    acc = np.sum(y_hat == y) / len(y)
    rt = time.time() - start_time
    return acc, y_hat, rt


def algo_fast_onc_k1(
        csr_mat, weighted, lambda_, y, nodes, eps):
    """ FastONL for kernel ID-1: \beta*I + D^{-1/2} W D^{-1/2} """
    start_time = time.time()
    n, _ = csr_mat.shape
    labels = [int(_) for _ in set(y)]
    k = len(labels) if -1 not in labels else len(labels) - 1
    assert len(nodes) == len(y)
    alpha = lambda_ / n
    gk = GraphKernel(csr_mat, alpha, eps, weighted)
    a = 0.
    g = np.zeros((k, n))
    y_hat = np.zeros(len(nodes))
    t = k * (n ** 2.)  # TODO one can use 2 *(n**2.)
    for ii, uu in enumerate(nodes):
        if int(y[uu]) == -1:  # ignore the unlabeled nodes
            continue
        v1 = gk.forward_push_m1(uu)
        re_vec = 2. * lambda_ * v1
        v = np.dot(g, re_vec)
        psi_t = -v / np.sqrt(a + k * t)
        qt, st = find_water_filling(psi_t)
        y_hat[uu] = int(np.argmax(qt))
        loss, grad = loss_grad_val(psi_t, int(y[uu]), st, loss_type='hinge')
        g[:, uu] = grad
        a += (2. * np.dot(grad, v) + re_vec[uu] * np.linalg.norm(grad) ** 2.)
        t -= re_vec[uu]
    acc = np.sum(y_hat == y) / len(y)
    rt = time.time() - start_time
    return acc, y_hat, rt


def algo_fast_onc_k2(
        csr_mat, weighted, lambda_, y, nodes, eps):
    """ FastONL for kernel ID-2: L (normalized graph laplacian)"""
    start_time = time.time()
    n, _ = csr_mat.shape
    labels = [int(_) for _ in set(y)]
    k = len(labels) if -1 not in labels else len(labels) - 1
    assert len(nodes) == len(y)
    alpha = lambda_ / (n + lambda_)
    gk = GraphKernel(csr_mat, alpha, eps, weighted)
    deg = csr_mat.sum(1).A.flatten()
    sqrt_deg = np.sqrt(deg)
    t = k * (n ** 2.)  # TODO one can use 2 *(n**2.)
    a = 0.
    g = np.zeros((k, n))
    y_hat = np.zeros(len(nodes))
    for ii, uu in enumerate(nodes):
        if int(y[uu]) == -1:  # ignore the unlabeled nodes
            continue
        v2 = gk.forward_push_m2(uu)
        re_vec = 2. * n * ((v2 / sqrt_deg) * sqrt_deg[uu])
        v = np.dot(g, re_vec)
        psi_t = -v / np.sqrt(a + k * t)
        qt, st = find_water_filling(psi_t)
        y_hat[uu] = int(np.argmax(qt))
        loss, grad = loss_grad_val(psi_t, int(y[uu]), st, loss_type='hinge')
        g[:, uu] = grad
        a += (2. * np.dot(grad, v) + re_vec[uu] * np.linalg.norm(grad) ** 2.)
        t -= re_vec[uu]
    acc = np.sum(y_hat == y) / len(y)
    rt = time.time() - start_time
    return acc, y_hat, rt


def algo_fast_onc_k3(
        csr_mat, weighted, lambda_, y, nodes, eps):
    """ FastONL for kernel ID-3: I - \beta D^{-1/2} W D^{-1/2} """
    start_time = time.time()
    n, _ = csr_mat.shape
    labels = [int(_) for _ in set(y)]
    k = len(labels) if -1 not in labels else len(labels) - 1
    assert len(nodes) == len(y)
    alpha = 1. - lambda_ / (n + lambda_)

    gk = GraphKernel(csr_mat, alpha, eps, weighted)
    deg = csr_mat.sum(1).A.flatten()
    sqrt_deg = np.sqrt(deg)
    a = 0.
    g = np.zeros((k, n))
    y_hat = np.zeros(len(nodes))
    t = k * (n ** 2.)  # TODO one can use 2 *(n**2.)
    for ii, uu in enumerate(nodes):
        if int(y[uu]) == -1:  # ignore the unlabeled nodes
            continue
        v3 = gk.forward_push_m2(uu)
        re_vec = 2. * lambda_ * ((v3 / sqrt_deg) * sqrt_deg[uu])
        v = np.dot(g, re_vec)
        psi_t = -v / np.sqrt(a + 2. * t)
        qt, st = find_water_filling(psi_t)
        y_hat[uu] = int(np.argmax(qt))
        loss, grad = loss_grad_val(psi_t, int(y[uu]), st, loss_type='hinge')
        g[:, uu] = grad
        a += (2. * np.dot(grad, v) + re_vec[uu] * np.linalg.norm(grad) ** 2.)
        t -= re_vec[uu]
    acc = np.sum(y_hat == y) / len(y)
    rt = time.time() - start_time
    return acc, y_hat, rt


def algo_fast_onc_k4(
        csr_mat, weighted, lambda_, y, nodes, eps):
    """ FastONL for kernel ID-4: \mathcal{L} = D - W
         (un-normalized graph laplacian)"""
    start_time = time.time()
    n, _ = csr_mat.shape
    labels = [int(_) for _ in set(y)]
    k = len(labels) if -1 not in labels else len(labels) - 1
    assert len(nodes) == len(y)
    alpha = 1.  # \alpha = \beta + \lambda/n, where \beta = 1 - \lambda/n
    gk = GraphKernel(csr_mat, alpha, eps, weighted)
    deg = csr_mat.sum(1).A.flatten()
    sqrt_deg = np.sqrt(deg)
    t = k * (n ** 2.)  # TODO one can use 2 *(n**2.)
    a = 0.
    g = np.zeros((k, n))
    y_hat = np.zeros(len(nodes))
    for ii, uu in enumerate(nodes):
        if int(y[uu]) == -1:  # ignore the unlabeled nodes
            continue

        v4 = gk.forward_push_m1(uu)
        # Note: in our original submission,
        # we directly use D^{1/2}*(alpha*I + D - A)^{-1}*D^{1/2}
        # re_vec = sqrt_deg * v4 * sqrt_deg
        # However, we find the following is more reasonable.
        re_vec = 2. * lambda_ * ((v4 / sqrt_deg) * sqrt_deg[uu])
        v = np.dot(g, re_vec)
        psi_t = -v / np.sqrt(a + k * t)
        qt, st = find_water_filling(psi_t)
        y_hat[uu] = int(np.argmax(qt))
        loss, grad = loss_grad_val(psi_t, int(y[uu]), st, loss_type='hinge')
        g[:, uu] = grad
        a += (2. * np.dot(grad, v) + re_vec[uu] * np.linalg.norm(grad) ** 2.)
        t -= re_vec[uu]
    acc = np.sum(y_hat == y) / len(y)
    rt = time.time() - start_time
    return acc, y_hat, rt


def algo_weighted_majority(
        csr_mat, y, nodes, seed, verbose=True):
    """
    Predicts via neighbors.
    Returns:
    """
    np.random.seed(seed)
    start_time = time.time()
    n, _ = csr_mat.shape
    labels = [int(_) for _ in set(y)]
    k = len(labels) if -1 not in labels else len(labels) - 1
    assert len(nodes) == len(y)
    y_pred = np.zeros(len(y), dtype=int)
    mark_labeled = -np.ones(n)
    indptr = csr_mat.indptr
    indices = csr_mat.indices
    data = csr_mat.data
    for ii, uu in enumerate(nodes):
        if int(y[uu]) == -1:  # ignore the unlabeled nodes
            continue
        prob_dist = np.zeros(k)
        for v_ind in range(indptr[uu], indptr[uu + 1]):
            vv = indices[v_ind]
            wei = data[v_ind]
            if mark_labeled[vv] != -1:
                prob_dist[int(y[vv])] += wei
        if np.sum(prob_dist) <= 0.:
            y_pred[uu] = int(np.random.choice(range(k)))
        else:
            y_pred[uu] = int(np.argmax(prob_dist))
        mark_labeled[uu] = 1
    if verbose:
        match = [y_pred[_] == y[_] for _ in nodes if y[_] != -1]
        acc = np.sum(match)
        print(f"Weighted Majority predict-right:{acc} accuracy:{acc / len(match)}")
    rt = time.time() - start_time
    return y_pred, rt


def algo_inverse_approx(
        csr_mat, kappa, y, nodes, num_iter=10, verbose=True):
    start_time = time.time()

    n, _ = csr_mat.shape
    labels = [int(_) for _ in set(y)]
    k = len(labels) if -1 not in labels else len(labels) - 1
    assert len(nodes) == len(y)

    coef = (1. / (2. * kappa) + 1. / (2. * n))
    diags = csr_mat.sum(axis=1).A.flatten()
    L = (n / (n + kappa)) * csr_mat
    with np.errstate(divide="ignore"):
        diags_sqrt = 1. / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], n, n, format="csr")
    KK = DH @ (L @ DH)
    YY = KK
    ZZ = sp.eye(n)
    for i in range(num_iter):
        ZZ += YY
        YY = YY @ KK
    m = coef * ZZ
    t = 2. * (n ** 2.)
    a = 0.
    g = np.zeros((k, n))
    y_pred = np.zeros(n)
    for ii, uu in enumerate(nodes):
        if int(y[uu]) == -1:  # ignore the unlabeled nodes
            continue
        m_vec = m.getrow(uu).todense()
        m_vec = np.squeeze(np.array(m_vec))
        v = np.dot(g, m_vec)
        psi_t = -v / np.sqrt(a + k * t)
        qt, st = find_water_filling(psi_t)
        y_pred[uu] = int(np.argmax(qt))
        loss, grad = loss_grad_val(psi_t, int(y[uu]), st, loss_type='hinge')
        g[:, uu] = grad
        a += (2. * np.dot(grad, v) + m_vec[uu] * np.linalg.norm(grad) ** 2.)
        t -= m_vec[uu]
    if verbose:
        match = [y_pred[_] == y[_] for _ in nodes if y[_] != -1]
        acc = np.sum(match)
        print(f"Inverse Approx predict-right:{acc} accuracy:{acc / len(match)} nnz:{m.nnz / n ** 2.}")
    rt = time.time() - start_time
    return y_pred, rt, m.nnz


def algo_online_rakhlin(
        csr_mat, kappa, y, nodes, mat_type='normalized'):
    start_time = time.time()

    n, _ = csr_mat.shape
    labels = [int(_) for _ in set(y)]
    k = len(labels) if -1 not in labels else len(labels) - 1
    assert len(nodes) == len(y)

    lap = laplacian_mat(adj_mat=csr_mat, lap_type=mat_type)
    m1 = (lap / (2. * kappa) + sp.eye(n) / (2. * n)).todense()
    m = np.linalg.inv(m1)
    t = np.trace(m)
    a = 0.
    g = np.zeros((k, n))
    y_pred = np.zeros(n)
    for ii, uu in enumerate(nodes):
        if int(y[uu]) == -1:  # ignore the unlabeled nodes
            continue
        v = np.dot(g, np.squeeze(np.asarray(m[uu])))
        psi_t = -v / np.sqrt(a + k * t)
        qt, st = find_water_filling(psi_t)
        y_pred[uu] = int(np.argmax(qt))
        loss, grad = loss_grad_val(psi_t, int(y[uu]), st, loss_type='hinge')
        g[:, uu] = grad
        a += (2. * np.dot(grad, v) + m[uu, uu] * np.linalg.norm(grad) ** 2.)
        t -= m[uu, uu]
    match = [y_pred[_] == y[_] for _ in nodes if y[_] != -1]
    acc = np.sum(match) / len(match)
    return acc, time.time() - start_time


def algo_weighted_tree(
        csr_mat, y, nodes, s, seed, verbose=True):
    import networkx as nx
    start_time = time.time()
    np.random.seed(seed)
    n, _ = csr_mat.shape
    assert len(nodes) == len(y)

    y_pred_s = np.zeros(shape=(s, n))
    assert len(y) == len(nodes)
    for i in range(s):
        root = np.random.choice(range(n))
        indptr = csr_mat.indptr
        indices = csr_mat.indices
        data = csr_mat.data
        rst, tree_edges = find_wilson_rst(n, indptr, indices, data, root)
        graph = nx.DiGraph()
        weights = dict()
        for uu, vv, wei in tree_edges:
            graph.add_edge(uu, vv, weight=wei)
            graph.add_edge(vv, uu, weight=wei)
            weights[(uu, vv)] = wei
            weights[(vv, uu)] = wei
        ell_prime = []
        for uu, vv, _ in nx.dfs_labeled_edges(graph, source=root):
            if uu == vv:
                continue
            if _ == 'forward':
                ell_prime.append((uu, vv, weights[(uu, vv)]))
            if _ == 'reverse':
                ell_prime.append((vv, uu, weights[(vv, uu)]))
        # -- from random spanning tree to unique-node tree
        visit = np.zeros(n, dtype=np.bool_)
        ell = [ell_prime[0]]
        visit[ell_prime[0][0]] = True
        visit[ell_prime[0][1]] = True
        ii = 1
        jj = 1
        while True:
            uu, vv, wei = ell_prime[ii]
            min_edge = [uu, vv, wei]
            for jj in range(ii, len(ell_prime)):
                uu, vv, wei = ell_prime[jj]
                if not visit[vv]:
                    visit[vv] = True
                    min_edge[1] = vv
                    break
                else:
                    if min_edge[2] > wei:
                        min_edge = [uu, vv, wei]
            ell.append(min_edge)
            if len(ell) == n - 1:
                break
            ii = jj
        ell_list = [ell[0][0], ell[0][1]]
        for edge in ell[1:]:
            ell_list.append(edge[1])
        # make prediction
        y_pred = np.zeros(len(nodes))
        y_pred[0] = 0  # make the first prediction
        dist = np.zeros(len(ell) + 1, dtype=float)
        for ii, edge in enumerate(ell):
            dist[ii + 1] = dist[ii] + 1. / edge[2]
        nodes_ind_map = dict()
        for ind, node in enumerate(ell_list):
            nodes_ind_map[node] = ind
        for ii in range(1, len(nodes)):
            uu = nodes[ii]
            if int(y[uu]) == -1:
                continue
            min_dist = np.infty
            min_ind = 0
            for jj in range(ii):
                vv = nodes[jj]
                uu_ind = nodes_ind_map[uu]
                vv_ind = nodes_ind_map[vv]
                d_ij = np.abs(dist[uu_ind] - dist[vv_ind])
                if min_dist > d_ij:
                    min_dist = d_ij
                    min_ind = nodes[jj]
            y_pred[uu] = y[min_ind]
        y_pred_s[i] = y_pred
        if verbose:
            match = [y_pred[_] == y[_] for _ in nodes if y[_] != -1]
            acc = np.sum(match)
            print(f"WTA-{i} predict-right:{acc} accuracy:{acc / n}")
    # majority vote
    y_pred = np.zeros_like(y)
    for i in range(len(nodes)):
        ll = list(y_pred_s[:, i])
        y_pred[i] = max(ll, key=ll.count)
    if verbose:
        match = [y_pred[_] == y[_] for _ in nodes if y[_] != -1]
        acc = np.sum(match)
        print(f"WTA-final predict-right:{acc} accuracy:{acc / n}")
    rt = time.time() - start_time
    return y_pred, rt

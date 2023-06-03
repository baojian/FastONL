import numpy as np
from algo import GraphKernel
from utils import load_graph_data
import matplotlib.pyplot as plt
import seaborn as sns


def draw_upper_bounds(dataset='pubmed', alpha=.2, num_eps=20, trials=10):
    csr_mat, labels = load_graph_data(dataset=dataset)
    np.random.seed(17)
    n, _ = csr_mat.shape
    m = len(csr_mat.data) / 2.
    list_ratios_1 = []
    list_ratios_2 = []
    list_ratios_3 = []
    list_active_aver_r = []
    list_aver_local_ratio_r = []
    list_total_iter = []
    list_total_true_iter = []
    list_num_operations = []
    true_bound = np.sqrt((1. - alpha) / (1. + alpha)) / n
    start = np.log10(0.01 * true_bound)
    end = np.log10(100 * true_bound)
    eps_list = np.logspace(start, end, num=num_eps, base=10.)
    print(f"n: {n} m:{m} eps-min:{eps_list[0]:7e} eps-max:{eps_list[-1]:7e} interesting-bound: {true_bound:7e}")
    for ind, eps in enumerate(eps_list):
        gk = GraphKernel(csr_mat, alpha, eps, weighted=False)
        s_active_aver_r = []
        s_aver_local_ratio_r = []
        s_estimate_t = []
        s_ratio_1 = []  # Our bound
        s_ratio_2 = []  # Sublinear bound
        s_ratio_3 = []  # Graph dependent bound
        s_num_operations = []
        s_t = []
        for s in np.random.permutation(n)[:trials]:
            sum_r, aver_active_vol, aver_ratio, t, estimate_t, num_operations, th_bound, _ = gk.forward_push_bounds(s)
            s_active_aver_r.append(aver_active_vol)
            s_aver_local_ratio_r.append(aver_ratio)
            s_ratio_1.append(min([th_bound, 1. / (alpha * eps)]))
            s_ratio_2.append(1. / (alpha * eps))
            s_estimate_t.append(estimate_t)
            s_num_operations.append(num_operations)
            s_t.append(t)

            if eps < 1. / (2. * m):
                s_ratio_3.append(((m / alpha) * np.log(1. / (m * eps)) + m))
        list_active_aver_r.append(np.mean(s_active_aver_r))
        list_aver_local_ratio_r.append(np.mean(s_aver_local_ratio_r))
        list_total_iter.append(np.mean(s_estimate_t))
        list_ratios_1.append(np.mean(s_ratio_1))
        list_ratios_2.append(np.mean(s_ratio_2))
        if len(s_ratio_3) > 0:
            list_ratios_3.append(np.mean(s_ratio_3))
        list_total_true_iter.append(np.mean(s_t))
        list_num_operations.append(np.mean(s_num_operations))
    plt.rc('font', family='serif', serif='Times New Roman')
    font = {'family': "Times New Roman",
            'weight': 'bold',
            'size': 18}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    sns.set()
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))

    ax[0].loglog(eps_list, list_ratios_2,
                 label=r"$\frac{1}{\epsilon \alpha}$", marker="D")
    ax[0].loglog(eps_list[:len(list_ratios_3)], list_ratios_3,
                 label=r"$\frac{m}{\alpha}\log(\frac{1}{\epsilon m}) + m$", marker="H")
    ax[0].loglog(eps_list, list_ratios_1,
                 label=r"$\frac{\operatorname{vol}(S_{1:T})}"
                       r"{\alpha \cdot \gamma_{1:T}}\log\left(\frac{1}{\epsilon (1-\alpha)|I_{T+1}|}\right)$",
                 c="r", marker="s")
    ax[0].loglog(eps_list, list_num_operations,
                 label=r"$R_T = \sum_{t=1}^T\operatorname{vol}(S_t)$", marker="o")
    ax[0].loglog(eps_list, [n] * len(eps_list), label=r"$n$", linestyle="--", c='b')
    ax[0].axvline(x=true_bound, color='gray', linestyle='dotted')

    ax[1].loglog(eps_list, [n] * len(eps_list), label=r"$n$", linestyle="--", c='b')
    ax[1].loglog(eps_list, [m] * len(eps_list), label=r"$m$", linestyle="dotted")
    ax[1].loglog(eps_list, list_active_aver_r,
                 label=r"$\operatorname{vol}(S_{1:T})$",
                 c="r", marker='s')
    ax[2].plot(eps_list, np.asarray(list_aver_local_ratio_r),
               label=r"$\gamma_{1:T}$", c="r", marker="s")
    ax[2].set_xscale('log')

    ax[3].plot(eps_list, np.asarray(list_total_true_iter), label=r"$T$", marker="D")
    ax[3].plot(eps_list, np.asarray(list_total_iter),
               label=r"$\frac{1}{\alpha \cdot \gamma_{1:T}}"
                     r"\log\left(\frac{1}{\epsilon (1-\alpha)|I_{T+1}|}\right)$", c="r", marker="s")
    ax[3].set_xscale('log')

    for i in range(4):
        ax[i].set_xlabel(r'$\epsilon$')
        ax[i].legend()
    ax[0].set_title(r'Number of Operations', fontsize=18)
    ax[1].set_title(r'Average volume of $S_t$', fontsize=18)
    ax[2].set_title(r'Linear Convergence factor', fontsize=18)
    ax[3].set_title(r'Total Epochs', fontsize=18)
    plt.subplots_adjust(wspace=0.16, hspace=0.1)
    fig.savefig(f"figs/fig-theoretical-upper-bound-alpha-{alpha}-data-{dataset}.pdf", dpi=400,
                bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


if __name__ == '__main__':
    draw_upper_bounds()
    for dataset in ['political-blog', 'citeseer', 'cora',
                    'pubmed', 'blogcatalog', 'flickr',
                    'youtube', 'ogbn-products', 'ogbn-arxiv']:
        for alpha in [0.2, 0.5, 0.8]:
            draw_upper_bounds(alpha=alpha, dataset=dataset)

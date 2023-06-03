import numpy as np
import seaborn as sns
from algo import GraphKernel
from matplotlib import pyplot as plt

from utils import load_graph_data


def draw_power_law(dataset="cora"):
    np.random.seed(183)
    plt.rcParams["font.size"] = 18
    cmap = sns.light_palette("seagreen", as_cmap=True)
    csr_mat, labels = load_graph_data(dataset=dataset)
    n, _ = csr_mat.shape
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=None, hspace=None)
    alpha_list = [0.1, 0.3, 0.5]
    for ind, alpha in enumerate(alpha_list):
        eps = 0.01 * np.sqrt((1 - alpha) / (1 + alpha)) / n
        gk = GraphKernel(csr_mat, alpha, eps, weighted=False)
        for s in np.random.permutation(n)[:10]:
            x_vec = gk.forward_push_m2(s)
            non_zeros = x_vec[np.nonzero(x_vec)]
            xx = np.asarray(range(1, len(non_zeros) + 1))
            yy = np.asarray(sorted(non_zeros, reverse=True))
            step = int(len(non_zeros) / 300)
            if step > 0:
                indices = range(0, len(xx), step)
            else:
                indices = range(0, len(xx))
            zz = ax[ind].scatter(xx[indices], yy[indices],
                                 marker="o", s=2, c=np.log10(yy[indices]), cmap=cmap, alpha=0.85)
            ax[ind].set_yscale('log')
            ax[ind].set_xscale('log')
    cbar = plt.colorbar(zz, ticks=[-1, -2, -3, -4, -5])
    cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^{-2}$',
                             r'$10^{-3}$', r'$10^{-4}$',
                             r'$10^{-5}$'], size=18)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(f"figs/fig-power-law-{dataset}.pdf", dpi=200,
                bbox_inches='tight', pad_inches=0.05, format='pdf')


for dataset in ['political-blog', 'citeseer', 'cora',
                'pubmed', 'blogcatalog', 'flickr',
                'youtube', 'ogbn-products', 'ogbn-arxiv']:
    draw_power_law(dataset=dataset)

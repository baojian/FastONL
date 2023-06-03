import numpy as np
import seaborn as sns
from algo import GraphKernel
import matplotlib as mpl
from matplotlib import pyplot as plt
import networkx as nx


def get_karate_graph(f_path='./datasets/karate.adjlist', start_index=0):
    clusters = {0: [11, 5, 6, 7, 17],
                1: [26, 25, 28, 29, 32, 10, 3],
                2: [14, 4, 13, 8, 2, 1, 20, 22, 18, 12],
                3: [27, 30, 24, 19, 23, 16, 15, 21, 33, 34, 31, 9]}
    edges = []
    with open(f_path) as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.rstrip().split(' ')
            if start_index == 0:
                for item in items[1:]:
                    edges.append((int(items[0]) - 1, int(item) - 1))
            else:
                for item in items[1:]:
                    edges.append((int(items[0]), int(item)))
    if start_index == 0:
        for cluster in clusters:
            clusters[cluster] = [_ - 1 for _ in clusters[cluster]]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph, clusters


def draw_karate_graph_ppr_vec():
    np.random.seed(183)
    graph, clusters = get_karate_graph(start_index=1)
    fig, ax = plt.subplots(1, 1, frameon=True, figsize=(6, 6))
    plt.subplots_adjust(wspace=None, hspace=None)
    fig.patch.set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_edges(graph, pos, width=0.5)

    n = 34
    eps = 1e-12
    weighted_graph = nx.Graph()
    for (uu, vv) in graph.edges:
        dist = np.linalg.norm(pos[uu] - pos[vv])
        weighted_graph.add_edge(uu - 1, vv - 1, weight=dist)
    csr_mat = nx.to_scipy_sparse_matrix(weighted_graph, nodelist=range(n))
    alpha = .85
    gk = GraphKernel(csr_mat, alpha, eps, weighted=True)
    node = 21
    for s in [node]:
        v1 = gk.forward_push_m1(s)
        v1[s] = 0
        v1[s] = np.max(v1)
    nx.draw_networkx_nodes(graph, pos, nodelist=clusters[0], node_size=250)
    nx.draw_networkx_nodes(graph, pos, nodelist=clusters[1], node_size=250)
    nx.draw_networkx_nodes(graph, pos, nodelist=clusters[2], node_size=250)
    nx.draw_networkx_nodes(graph, pos, nodelist=clusters[3], node_size=250)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='k')
    d = {_: v1[_] for _ in range(n)}
    low, *_, high = sorted(d.values())
    c_map = sns.light_palette("seagreen", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=c_map)

    nx.draw_networkx_nodes(graph, pos, nodelist=[i + 1 for i in range(n)],
                           node_color=[mapper.to_rgba(_) for _ in d.values()],
                           node_size=250)
    plt.savefig(f'figs/karate-graph-ppr-vec-{node + 1}-type-i.pdf',
                dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def draw_karate_graph_input():
    np.random.seed(183)
    clrs = sns.color_palette("husl", 4)
    graph, clusters = get_karate_graph(start_index=1)

    fig, ax = plt.subplots(1, 1, frameon=True, figsize=(6, 6))
    plt.subplots_adjust(wspace=None, hspace=None)
    fig.patch.set_visible(False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, nodelist=clusters[0], node_color=clrs[0], node_size=250)
    nx.draw_networkx_nodes(graph, pos, nodelist=clusters[1], node_color=clrs[1], node_size=250)
    nx.draw_networkx_nodes(graph, pos, nodelist=clusters[2], node_color=clrs[2], node_size=250)
    nx.draw_networkx_nodes(graph, pos, nodelist=clusters[3], node_color=clrs[3], node_size=250)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='k')
    nx.draw_networkx_edges(graph, pos, width=0.5)
    plt.savefig('figs/karate-graph.pdf', dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def draw_power_law():
    # I tried many times, this seed makes the most beautiful one.
    np.random.seed(183)
    plt.rcParams["font.size"] = 18
    cmap = sns.light_palette("seagreen", as_cmap=True)
    graph, clusters = get_karate_graph(start_index=1)
    pos = nx.spring_layout(graph)
    n = 34
    eps = 1e-12
    weighted_graph = nx.Graph()
    for (uu, vv) in graph.edges:
        dist = np.linalg.norm(pos[uu] - pos[vv])
        weighted_graph.add_edge(uu - 1, vv - 1, weight=dist)
    csr_mat = nx.to_scipy_sparse_matrix(weighted_graph, nodelist=range(n))
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(wspace=None, hspace=None)
    yy_s = []
    for alpha in np.arange(0.2, 0.21, 0.01):
        gk = GraphKernel(csr_mat, alpha, eps, weighted=True)
        for s in range(34):
            v1 = gk.forward_push_m2(s)
            xx = range(1, 35)
            yy = sorted(v1, reverse=True)
            zz = ax.scatter(xx, yy, marker="o", s=50, c=np.log10(yy),
                            cmap=cmap, alpha=0.85)
            ax.set_yscale('log')
            ax.set_xscale('log')
            for y in yy:
                yy_s.append(y)
    min_yy, max_yy = np.log10(np.min(yy_s)), np.log10(np.max(yy_s))
    print(min_yy, max_yy)
    range_xx = (max_yy - min_yy) / 4.1
    print(np.arange(min_yy, max_yy, range_xx))
    xxx = [-2.73, -2.17799107, -1.62972612, -1.08146117, -0.53319623]
    cbar = plt.colorbar(zz, ticks=xxx[::-1])
    cbar.ax.set_yticklabels([r'$10^{-0.5}$', r'$10^{-1.1}$',
                             r'$10^{-1.6}$', r'$10^{-2.2}$',
                             r'$10^{-2.7}$'], size=18)
    ax.set_ylabel('$x_{s,\epsilon}(i)$')
    ax.set_xlabel('Ranking of $x_{s,\epsilon}(i)$')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(f"figs/fig-karate-graph-power-law-all.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0.05, format='pdf')


def draw_power_law_m1():
    np.random.seed(183)
    plt.rcParams["font.size"] = 18
    cmap = sns.light_palette("seagreen", as_cmap=True)
    graph, clusters = get_karate_graph(start_index=1)
    pos = nx.spring_layout(graph)
    n = 34
    eps = 1e-12
    weighted_graph = nx.Graph()
    for (uu, vv) in graph.edges:
        dist = np.linalg.norm(pos[uu] - pos[vv])
        weighted_graph.add_edge(uu - 1, vv - 1, weight=dist)
    csr_mat = nx.to_scipy_sparse_matrix(weighted_graph, nodelist=range(n))
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(wspace=None, hspace=None)
    yy_s = []
    for alpha in np.arange(0.85, 0.86, 0.01):
        gk = GraphKernel(csr_mat, alpha, eps, weighted=True)
        for s in range(34):
            v1 = gk.forward_push_m1(s)
            xx = range(1, 35)
            yy = sorted(v1, reverse=True)
            zz = ax.scatter(xx, yy, marker="o", s=50, c=np.log10(yy),
                            cmap=cmap, alpha=0.85)
            ax.set_yscale('log')
            ax.set_xscale('log')
            for y in yy:
                yy_s.append(y)
    min_yy, max_yy = np.log10(np.min(yy_s)), np.log10(np.max(yy_s))
    print(min_yy, max_yy)
    range_xx = (max_yy - min_yy) / 4.1
    print(np.arange(min_yy, max_yy, range_xx))
    xxx = [-2.73, -2.17799107, -1.62972612, -1.08146117, -0.53319623]
    cbar = plt.colorbar(zz, ticks=xxx[::-1])
    cbar.ax.set_yticklabels([r'$10^{-0.5}$', r'$10^{-1.1}$',
                             r'$10^{-1.6}$', r'$10^{-2.2}$',
                             r'$10^{-2.7}$'], size=18)
    ax.set_ylabel('$x_{s,\epsilon}(i)$')
    ax.set_xlabel('Ranking of $x_{s,\epsilon}(i)$')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(f"figs/fig-karate-graph-power-law-type-i.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0.05, format='pdf')


if __name__ == '__main__':
    draw_karate_graph_input()
    draw_karate_graph_ppr_vec()
    draw_power_law_m1()

import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt


def get_fast_onc(dataset_list, eps='5e-5', method='fast-onc'):
    alpha_list = np.arange(0.05, 1.01, 0.05)
    trials = 20
    error_rate_mat = dict()
    for dataset in dataset_list:
        print(f'process {method} {dataset}')
        results = pkl.load(open(f'baselines/results_{method}_eps-{eps}_{dataset}.pkl', 'rb'))
        for (trial_i, para_ind, method_v), (rt, y, y_pred, nodes) in results.items():
            if (dataset, method_v, para_ind) not in error_rate_mat:
                error_rate_mat[(dataset, method_v, para_ind)] = [0] * trials
            match = np.asarray([y_pred[_] == y[_] for _ in nodes if y[_] != -1])
            error_rate = 1. - np.cumsum(match) / np.arange(1, len(match) + 1)
            error_rate_mat[(dataset, method_v, para_ind)][trial_i] = error_rate

    error_rate_mean_mat = dict()
    error_rate_std_mat = dict()
    for (dataset, method_v, para_ind) in error_rate_mat:
        error_rate_mean_mat[(dataset, method_v, para_ind)] = np.mean(
            error_rate_mat[(dataset, method_v, para_ind)], axis=0)
    for (dataset, method_v, para_ind) in error_rate_mat:
        error_rate_std_mat[(dataset, method_v, para_ind)] = np.std(
            error_rate_mat[(dataset, method_v, para_ind)], axis=0)
    pkl.dump([error_rate_mean_mat, error_rate_std_mat, alpha_list],
             open(f'baselines/fig_fast-onc_{method}_eps-{eps}.pkl', 'wb'))
    return error_rate_mean_mat, error_rate_std_mat, alpha_list


def main():
    eps = '5e-5'
    method = 'fast-onc'
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10', 'blogcatalog']
    dataset_title_list = ['Political', 'Citeseer', 'Cora',
                          'Pudmed', 'MNIST', 'Blogcatalog']
    re_mean, re_std, alpha_list = pkl.load(open(f'baselines/fig_fast-onc_{method}_eps-{eps}.pkl', 'rb'))
    plt.rc('font', family='serif', serif='Times New Roman')
    font = {'family': "Times New Roman",
            'weight': 'bold',
            'size': 20}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    sns.set()
    clrs = sns.color_palette()
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        ii = int(i / 3)
        jj = i % 3
        xx = alpha_list
        print(ii, jj)
        yy = [1 - re_mean[(dataset, 'fast-onc-v1', _)][-1] for _ in range(20)]
        ax[ii, jj].plot(xx, yy, label=r'$\textsc{Fast-ONC}_{\mathcal{M}_1}$',
                        c=clrs[0], linewidth=1.5, marker="D", markersize=4)
        yy = [1 - re_mean[(dataset, 'fast-onc-v2', _)][-1] for _ in range(20)]
        ax[ii, jj].plot(xx, yy, label=r'$\textsc{Fast-ONC}_{\mathcal{M}_2}$',
                        c=clrs[1], linewidth=1.5, marker="s", markersize=4)
        ax[ii, jj].set_title(dataset_title_list[i])
        ax[1, jj].set_xlabel(r"$\alpha$", fontsize=18)
        ax[0, jj].set_xticks([])
        ax[ii, 0].set_ylabel(r"Accuracy", fontsize=18)
    plt.subplots_adjust(wspace=0.2, hspace=0.15)
    fig.savefig(f"figs/fig3-fast-onc.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


if __name__ == '__main__':
    main()

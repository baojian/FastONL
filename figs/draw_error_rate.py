import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt


def get_fast_onc(dataset_list, eps='5e-5', method='fast-onc'):
    trials = 20
    error_rate_mat = dict()
    for dataset in dataset_list:
        print(f'process {method} {dataset}')
        results = pkl.load(open(f'baselines/results_{method}_eps-{eps}_{dataset}.pkl', 'rb'))
        for (trial_i, para_ind, method_v), (rt, y, y_pred, nodes) in results.items():
            if (dataset, method_v, para_ind) not in error_rate_mat:
                error_rate_mat[(dataset, method_v, para_ind)] = [0] * 20
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

    for (dataset, method_v) in [(_, 'fast-onc-v1') for _ in dataset_list]:
        best_one = None
        best_val = np.infty
        best_std = None
        for i in range(trials):
            val = error_rate_mean_mat[(dataset, method_v, i)][-1]
            print(method_v, best_val, val)
            if val < best_val:
                best_val = val
                best_one = error_rate_mean_mat[(dataset, method_v, i)]
                best_std = error_rate_std_mat[(dataset, method_v, i)]
        error_rate_mean_mat[(dataset, method_v)] = best_one
        error_rate_std_mat[(dataset, method_v)] = best_std
    for (dataset, method_v) in [(_, 'fast-onc-v2') for _ in dataset_list]:
        best_one = None
        best_val = np.infty
        best_std = None
        for i in range(trials):
            val = error_rate_mean_mat[(dataset, method_v, i)][-1]
            print(method_v, best_val, val)
            if val < best_val:
                best_val = val
                best_one = error_rate_mean_mat[(dataset, method_v, i)]
                best_std = error_rate_std_mat[(dataset, method_v, i)]
        error_rate_mean_mat[(dataset, method_v)] = best_one
        error_rate_std_mat[(dataset, method_v)] = best_std
    re_mean = dict()
    re_std = dict()
    for key in error_rate_mean_mat:
        if len(key) == 2:
            re_mean[key] = error_rate_mean_mat[key]
            re_std[key] = error_rate_std_mat[key]
    pkl.dump([re_mean, re_std], open(f'baselines/fig_error-rate_{method}_eps-{eps}.pkl', 'wb'))
    return re_mean, re_std


def get_inverse_approx(dataset_list, method='inverse-approx', num_iter=5):
    trials = 20
    error_rate_mat = dict()
    error_rate_mean_mat = dict()
    error_rate_std_mat = dict()
    for dataset in dataset_list:
        print(f'process {method} {dataset}')
        results = pkl.load(open(f'baselines/results_{method}_num-iter-{num_iter}_{dataset}.pkl', 'rb'))
        for (trial_i, para_ind, method_v), (rt, nnz, y, y_pred, nodes) in results.items():
            if (dataset, method_v, para_ind) not in error_rate_mat:
                error_rate_mat[(dataset, method_v, para_ind)] = [0] * trials
            match = np.asarray([y_pred[_] == y[_] for _ in nodes if y[_] != -1])
            error_rate = 1. - np.cumsum(match) / np.arange(1, len(match) + 1)
            error_rate_mat[(dataset, method_v, para_ind)][trial_i] = error_rate
    for (dataset, method_v, para_ind) in error_rate_mat:
        error_rate_mean_mat[(dataset, method_v, para_ind)] = np.mean(
            error_rate_mat[(dataset, method_v, para_ind)], axis=0)
        error_rate_std_mat[(dataset, method_v, para_ind)] = np.std(
            error_rate_mat[(dataset, method_v, para_ind)], axis=0)
    for (dataset, method_v) in [(_, 'inverse-approx') for _ in dataset_list]:
        best_one = None
        best_val = np.infty
        best_std = None
        for i in range(trials):
            val = error_rate_mean_mat[(dataset, method_v, i)][-1]
            print(method_v, best_val, val)
            if val < best_val:
                best_val = val
                best_one = error_rate_mean_mat[(dataset, method_v, i)]
                best_std = error_rate_std_mat[(dataset, method_v, i)]
        error_rate_mean_mat[(dataset, method_v)] = best_one
        error_rate_std_mat[(dataset, method_v)] = best_std
    re_mean = dict()
    re_std = dict()
    for key in error_rate_mean_mat:
        if len(key) == 2:
            re_mean[key] = error_rate_mean_mat[key]
            re_std[key] = error_rate_std_mat[key]
    pkl.dump([re_mean, re_std], open(f'baselines/fig_error-rate_{method}_num-iter-{num_iter}.pkl', 'wb'))
    return re_mean, re_std


def get_weighted_majority(dataset_list, method='weighted-majority'):
    trials = 20
    error_rate_mat = dict()
    error_rate_mean_mat = dict()
    error_rate_std_mat = dict()
    for dataset in dataset_list:
        print(f'process {method} {dataset}')
        results = pkl.load(open(f'baselines/results_{method}_{dataset}.pkl', 'rb'))
        for (trial_i, method_v), (rt, y, y_pred, nodes) in results.items():
            if (dataset, method_v) not in error_rate_mat:
                error_rate_mat[(dataset, method_v)] = [0] * trials
            match = np.asarray([y_pred[_] == y[_] for _ in nodes if y[_] != -1])
            error_rate = 1. - np.cumsum(match) / np.arange(1, len(match) + 1)
            error_rate_mat[(dataset, method_v)][trial_i] = error_rate
    for dataset in dataset_list:
        error_rate_std_mat[(dataset, method)] = np.std(error_rate_mat[(dataset, method)], axis=0)
        error_rate_mean_mat[(dataset, method)] = np.mean(error_rate_mat[(dataset, method)], axis=0)
    pkl.dump([error_rate_mean_mat, error_rate_std_mat], open(f'baselines/fig_error-rate_{method}.pkl', 'wb'))
    return [error_rate_mean_mat, error_rate_std_mat]


def get_wta(dataset_list, method='wta', s=10):
    trials = 20
    error_rate_mat = dict()
    error_rate_mean_mat = dict()
    error_rate_std_mat = dict()
    for dataset in dataset_list:
        print(f'process {method} {dataset}')
        results = pkl.load(open(f'baselines/results_{method}_s-{s}_{dataset}.pkl', 'rb'))
        for (trial_i, method_v), (rt, y, y_pred, nodes) in results.items():
            if (dataset, method_v) not in error_rate_mat:
                error_rate_mat[(dataset, method_v)] = [0] * trials
            match = np.asarray([y_pred[_] == y[_] for _ in nodes if y[_] != -1])
            error_rate = 1 - np.cumsum(match) / np.arange(1, len(match) + 1)
            error_rate_mat[(dataset, method_v)][trial_i] = error_rate
    for dataset in dataset_list:
        error_rate_std_mat[(dataset, method)] = np.std(error_rate_mat[(dataset, method)], axis=0)
        error_rate_mean_mat[(dataset, method)] = np.mean(error_rate_mat[(dataset, method)], axis=0)
    pkl.dump([error_rate_mean_mat, error_rate_std_mat],
             open(f'baselines/fig_error-rate_{method}_s-{s}.pkl', 'wb'))
    return error_rate_mean_mat, error_rate_std_mat


def get_relaxation(dataset_list, method='relaxation', mat_type='normalized'):
    trials = 20
    error_rate_mat = dict()
    error_rate_mean_mat = dict()
    error_rate_std_mat = dict()
    for dataset in dataset_list:
        print(f'process {method} {dataset}')
        results = pkl.load(open(f'baselines/results_{method}-{mat_type}_{dataset}.pkl', 'rb'))
        for (trial_i, para_ind, method_v), (rt, y, y_pred, nodes) in results.items():
            if (dataset, method_v, para_ind) not in error_rate_mat:
                error_rate_mat[(dataset, method_v, para_ind)] = [0] * trials
            match = np.asarray([y_pred[_] == y[_] for _ in nodes if y[_] != -1])
            error_rate = 1. - np.cumsum(match) / np.arange(1, len(match) + 1)
            error_rate_mat[(dataset, method_v, para_ind)][trial_i] = error_rate
    for (dataset, method_v, para_ind) in error_rate_mat:
        error_rate_mean_mat[(dataset, method_v, para_ind)] = np.mean(
            error_rate_mat[(dataset, method_v, para_ind)], axis=0)
        error_rate_std_mat[(dataset, method_v, para_ind)] = np.std(
            error_rate_mat[(dataset, method_v, para_ind)], axis=0)
    for (dataset, method_v) in [(_, 'relaxation') for _ in dataset_list]:
        best_one = None
        best_val = np.infty
        best_std = None
        for i in range(trials):
            val = error_rate_mean_mat[(dataset, method_v, i)][-1]
            print(method_v, best_val, val)
            if val < best_val:
                best_val = val
                best_one = error_rate_mean_mat[(dataset, method_v, i)]
                best_std = error_rate_std_mat[(dataset, method_v, i)]
        error_rate_mean_mat[(dataset, method_v)] = best_one
        error_rate_std_mat[(dataset, method_v)] = best_std
    re_mean = dict()
    re_std = dict()
    for key in error_rate_mean_mat:
        if len(key) == 2:
            re_mean[key] = error_rate_mean_mat[key]
            re_std[key] = error_rate_std_mat[key]
    pkl.dump([re_mean, re_std], open(f'baselines/fig_error-rate_{method}.pkl', 'wb'))
    return re_mean, re_std


def generate_error_rate():
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10', 'blogcatalog']
    get_inverse_approx(dataset_list)
    get_fast_onc(dataset_list)
    get_wta(dataset_list)
    get_weighted_majority(dataset_list)

def get_all_error_rate():
    mean_mat1, std_mat1 = pkl.load(open('baselines/fig_error-rate_fast-onc_eps-5e-5.pkl', 'rb'))
    mean_mat2, std_mat2 = pkl.load(open('baselines/fig_error-rate_inverse-approx_num-iter-5.pkl', 'rb'))
    mean_mat3, std_mat3 = pkl.load(open('baselines/fig_error-rate_weighted-majority.pkl', 'rb'))
    mean_mat4, std_mat4 = pkl.load(open('baselines/fig_error-rate_wta_s-10.pkl', 'rb'))
    mean_mat5, std_mat5 = pkl.load(open('baselines/fig_error-rate_relaxation.pkl', 'rb'))
    mean_error_rate = {**mean_mat1, **mean_mat2, **mean_mat3, **mean_mat4, **mean_mat5}
    std_error_rate = {**std_mat1, **std_mat2, **std_mat3, **std_mat4, **std_mat5}
    return mean_error_rate, std_error_rate


def draw_error_rate():
    plt.rc('font', family='serif', serif='Times New Roman')
    font = {'family': "Times New Roman",
            'weight': 'bold',
            'size': 20}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    sns.set()
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.5))
    for axx in ax:
        axx.grid(b=True, which='major')
        axx.grid(b=True, which='minor')
    dataset_list = ['political-blog', 'citeseer', 'cora', 'pubmed',
                    'mnist-tr-nei10', 'blogcatalog']
    dataset_title_list = ['Political', 'Citeseer', 'Cora', 'Pudmed', 'MNIST']
    method_list = ['weighted-majority', 'wta', 'inverse-approx', 'relaxation', r'fast-onc-v1', r'fast-onc-v2']
    label_list = [r'WeightedMajority', r'WTA', r'InverseApprox', 'Relaxation',
                  r'$\textsc{Fast-ONC}_{\mathcal{M}_1}$', r'$\textsc{Fast-ONC}_{\mathcal{M}_2}$']
    mean_error_rate, std_error_rate = get_all_error_rate()
    clrs = sns.color_palette()
    start_ind = 20
    for ii, dataset in enumerate(dataset_list):
        for jj, method in enumerate(method_list):
            print(dataset, method)
            if (dataset, method) not in mean_error_rate:
                continue
            size = len(mean_error_rate[(dataset, method)])
            step = int(size / 300)
            indices = [start_ind + _ for _ in range(size) if (_ % step == 0) and (start_ind + _) < (size - 1)]
            if indices[-1] != size - 1:
                indices.append(size - 1)
            mean = mean_error_rate[(dataset, method)][indices]
            std = std_error_rate[(dataset, method)][indices]
            ax[ii].plot(indices, mean, label=label_list[jj], c=clrs[jj], linewidth=1.5)
            ax[ii].fill_between(indices, mean - std, mean + std, alpha=0.35, facecolor=clrs[jj])
        ax[ii].set_title(dataset_title_list[ii], fontsize=18)
        ax[ii].set_xlabel(r'Samples Seen')
    ax[0].set_xticks([200, 500, 800, 1100])
    ax[0].set_xticklabels(['200', '500', '800', '1100'])

    ax[1].set_xticks([200, 800, 1400, 2000])
    ax[1].set_xticklabels(['200', '800', '1400', '2000'])

    ax[2].set_xticks([300, 1000, 1700, 2400])
    ax[2].set_xticklabels(['300', '1000', '1700', '2400'])

    ax[3].set_xticks([2000, 7000, 12000, 17000])
    ax[3].set_xticklabels(['2000', '7000', '12000', '17000'])

    ax[4].set_xticks([1500, 5000, 8500, 12000])
    ax[4].set_xticklabels(['1500', '5000', '8500', '12000'])

    ax[0].set_ylabel(r'Error rate', fontsize=20)
    ax[0].legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.01)
    fig.savefig(f"figs/fig1-error-rate.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


if __name__ == '__main__':
    generate_error_rate()
    draw_error_rate()

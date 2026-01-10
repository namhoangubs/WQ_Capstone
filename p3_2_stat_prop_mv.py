from keras.utils import set_random_seed
from numpy import seterr, corrcoef, argmax, zeros, where, nan
from p2_0_utils import *
from matplotlib.pyplot import figure, hist, axvline, xlabel, ylabel, gcf, close, scatter, xscale, yscale
from numpy.linalg import eig
from seaborn import heatmap
from networkx import from_numpy_array, selfloop_edges, minimum_spanning_tree
from pandas import Series
from pathlib import Path


from pathlib import Path

BASE_DIR = Path(r"C:\Users\namho\Desktop\Previous PC\World Quant\Thesis\Replication")
INPUT_DIR = BASE_DIR / "Inputs"

def p3_2_stat_prop_mv(n_states=1, n_stocks=8):
    SEED = 42
    TRAIN_DATE = '2009-06-02'
    FOLDER = BASE_DIR / 'stylized_facts'

    set_random_seed(SEED)
    seterr(divide='ignore', invalid='ignore')

    stock_rets = read_csv(BASE_DIR / f'phase_2/stock_rets_q{n_states}_n{n_stocks}.csv', index_col=[0], parse_dates=[0])
    sim_rets = read_csv(BASE_DIR / f'simulations/sim_rets_q{n_states}_n{n_stocks}.csv', index_col=[0], parse_dates=[0])

    n_train = sum(stock_rets.index < TRAIN_DATE)
    real_rets = stock_rets[n_train:].drop('STATE', axis=1)

    print('F:  6 of 10')
    off_diag_rets_corrs = list()
    for rets in [real_rets, sim_rets]:
        rets_corr = corrcoef(rets, rowvar=False)
        rets_corr = rets_corr.flatten().round(5)
        off_diag_rets_corrs.append(rets_corr[rets_corr != 1.].tolist())

    plt_corr = list()
    for off_diag_rets_corr in off_diag_rets_corrs:
        figure(figsize=(4, 3))
        hist(off_diag_rets_corr, n_stocks, density=True)
        axvline(mean(off_diag_rets_corr), lw=0.5, c='k')
        axvline(lw=0.5, ls='--', c='k')
        xlabel('Correlation Coefficients')
        ylabel('Density')
        plt_corr.append(gcf())
        close()

    print('F:  7 of 10')
    rets_eig_vals_list = list()
    for rets in [real_rets, sim_rets]:
        rets_corr = corrcoef(rets, rowvar=False)
        eig_vals, _ = eig(rets_corr)
        rets_eig_vals_list.append(sorted(eig_vals, reverse=True))

    plt_eig_vals = list()
    for rets_eig_vals in rets_eig_vals_list:
        figure(figsize=(4, 3))
        hist(rets_eig_vals, n_stocks, density=True)
        xlabel('Eigenvalues')
        ylabel('Density')
        plt_eig_vals.append(gcf())
        close()

    print('F:  8 of 10')
    rets_eig_vecs_list = list()
    for rets in [real_rets, sim_rets]:
        rets_corr = corrcoef(rets, rowvar=False)
        eig_vals, eig_vecs = eig(rets_corr)
        max_rets_eig_vecs = (eig_vecs[:, argmax(eig_vals)])

        if all(max_rets_eig_vecs < 0):
            max_rets_eig_vecs = -max_rets_eig_vecs
        rets_eig_vecs_list.append(max_rets_eig_vecs)

    plt_eig_vecs = list()
    for rets_eig_vecs in rets_eig_vecs_list:
        figure(figsize=(4, 3))
        hist(rets_eig_vecs, n_stocks, density=True)
        axvline(mean(rets_eig_vecs), lw=0.5, c='k')
        axvline(ls='--', lw=0.5, c='k')
        xlabel('Dominant Eigenvector Values')
        ylabel('Density')
        plt_eig_vecs.append(gcf())
        close()

    print('F:  9 of 10')
    plt_h_clusts = list()
    for rets in [real_rets, sim_rets]:
        rets_corr = corrcoef(rets, rowvar=False)
        figure(figsize=(4, 3))
        heatmap(rets_corr, xticklabels=False, yticklabels=False)
        plt_h_clusts.append(gcf())
        close()

    print('F: 10 of 10')
    rets_node_degree_list = list()
    for rets in [real_rets, sim_rets]:
        rets_corr = corrcoef(rets, rowvar=False)
        graph = from_numpy_array(rets_corr)
        graph.remove_edges_from(selfloop_edges(graph))
        mst = minimum_spanning_tree(graph)

        degree = {i_stock: 0 for i_stock in range(n_stocks)}
        for edge in mst.edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        degree = Series(degree).sort_values(ascending=False)
        degree_counts = degree.value_counts()

        counts = zeros(sum(degree_counts))
        for node in range(sum(degree_counts)):
            if node in degree_counts:
                counts[node] = degree_counts[node]

        density = counts / sum(counts)
        density = where(density == 0, nan, density)
        rets_node_degree_list.append(density)

    plt_node_degree = list()
    for rets_node_degree in rets_node_degree_list:
        figure(figsize=(4, 3))
        scatter(range(sum(degree_counts)), rets_node_degree)
        xscale('log')
        yscale('log')
        xlabel('Node Degrees')
        ylabel('Density')
        plt_node_degree.append(gcf())
        close()

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    for i_rets, n_rets in iter(enumerate(['r', 's'])):
        plt_corr[i_rets].savefig(FOLDER / f'plt_{n_rets}_mv_c_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_eig_vals[i_rets].savefig(FOLDER / f'plt_{n_rets}_mv_val_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_eig_vecs[i_rets].savefig(FOLDER / f'plt_{n_rets}_mv_vec_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_h_clusts[i_rets].savefig(FOLDER / f'plt_{n_rets}_mv_hc_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_node_degree[i_rets].savefig(FOLDER / f'plt_{n_rets}_mv_nd_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p3_2_stat_prop_mv()
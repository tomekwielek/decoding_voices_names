import mne
import numpy as np
import os
import pandas as pd
from utils import pickle_load
from config import (get_path, myload, mysave, base_path_data, sleep_path_data, wake_event_id, sleep_event_id)
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt 
import re 
from scipy.stats import sem
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test
from scipy import stats as stats
import matplotlib.patches as mpatches
import matplotlib

matplotlib.rcParams.update({'font.size': 14})
#DEFINE
nsample = 105
dname = 'N2trained_ordered{}'.format(nsample)

#save_path = r'D:\SON_deep\results\decod\cross\tmp\voices\{}'.format(dname)
save_path = r'D:\SON_deep\results\decod\cross\tmp\names\{}'.format(dname)

sbjs = os.listdir(base_path_data)
sbjs = [sbjs[i] for i in range(len(sbjs)) if sbjs[i].startswith('VP')]

# # VOICES
# if dname == 'N1trained_ordered{}'.format(nsample):
#     contrast1 = ('familiar/N1', 'unfamiliar/N1')
#     contrast2 = ('familiar/N2', 'unfamiliar/N2')
#     contrast3 = ('familiar/N3', 'unfamiliar/N3')
#     contrast4 = ('familiar/REM', 'unfamiliar/REM')
#     contrast5 = ('familiar', 'unfamiliar') #wake

# elif dname == 'N2trained_ordered{}'.format(nsample):
#     #N2
#     contrast1 = ('familiar/N2', 'unfamiliar/N2')
#     contrast2 = ('familiar/N1', 'unfamiliar/N1')
#     contrast3 = ('familiar/N3', 'unfamiliar/N3')
#     contrast4 = ('familiar/REM', 'unfamiliar/REM')
#     contrast5 = ('familiar', 'unfamiliar') #wake

# elif dname == 'N3trained_ordered{}'.format(nsample):
#     #N3
#     contrast1 = ('familiar/N3', 'unfamiliar/N3')
#     contrast2 = ('familiar/N1', 'unfamiliar/N1')
#     contrast3 = ('familiar/N2', 'unfamiliar/N2')
#     contrast4 = ('familiar/REM', 'unfamiliar/REM')
#     contrast5 = ('familiar', 'unfamiliar') #wake

# elif dname == 'REMtrained_ordered{}'.format(nsample):
#     #REM
#     contrast1 = ('familiar/REM', 'unfamiliar/REM')
#     contrast2 = ('familiar/N1', 'unfamiliar/N1')
#     contrast3 = ('familiar/N2', 'unfamiliar/N2')
#     contrast4 = ('familiar/N3', 'unfamiliar/N3')
#     contrast5 = ('familiar', 'unfamiliar')
# elif dname == 'waketrained_ordered{}'.format(nsample):
#     # Wake trained 
#     contrast1 = ('familiarW', 'unfamiliarW')
#     contrast2 = ('familiar/N1', 'unfamiliar/N1')
#     contrast3 = ('familiar/N2', 'unfamiliar/N2')
#     contrast4 = ('familiar/N3', 'unfamiliar/N3')
#     contrast5 = ('familiar/REM', 'unfamiliar/REM')
# NAMES
if dname == 'N2trained_ordered{}'.format(nsample):
    #N2
    contrast1 = ('own/N2', 'un/N2')
    contrast2 = ('own/N1', 'un/N1')
    contrast3 = ('own/N3', 'un/N3')
    contrast4 = ('own/REM', 'un/REM')
    contrast5 = ('own', 'un') #wake

elif dname == 'N1trained_ordered{}'.format(nsample):
    contrast1 = ('own/N1', 'un/N1')
    contrast2 = ('own/N2', 'un/N2')
    contrast3 = ('own/N3', 'un/N3')
    contrast4 = ('own/REM', 'un/REM')
    contrast5 = ('own', 'un') #wake

elif dname == 'N3trained_ordered{}'.format(nsample):
    #N3
    contrast1 = ('own/N3', 'un/N3')
    contrast2 = ('own/N1', 'un/N1')
    contrast3 = ('own/N2', 'un/N2')
    contrast4 = ('own/REM', 'un/REM')
    contrast5 = ('own', 'un') #wake
elif dname == 'REMtrained_ordered{}'.format(nsample):
    #REM
    contrast1 = ('own/REM', 'un/REM')
    contrast2 = ('own/N1', 'un/N1')
    contrast3 = ('own/N2', 'un/N2')
    contrast4 = ('own/N3', 'un/N3')
    contrast5 = ('own', 'un') 
elif dname == 'waketrained_ordered{}'.format(nsample):
    # Wake trained 
    contrast1 = ('ownW', 'unknownW')
    contrast2 = ('own/N1', 'unknown/N1')
    contrast3 = ('own/N2', 'unknown/N2')
    contrast4 = ('own/N3', 'unknown/N3')
    contrast5 = ('own/REM', 'unknown/REM')

def mass_univ_corrected(X):
    from scipy.stats import mannwhitneyu
    from mne.stats import fdr_correction
    X0 = np.zeros(X.shape)
    X_flat = X.reshape(X.shape[0], -1)
    X0_flat = X0.reshape(X0.shape[0], -1)
    pvals = np.zeros(X_flat.shape[-1])
    for i in range(len(pvals)):
        stat, p = mannwhitneyu(X_flat[:,i], X0_flat[:,i], alternative='greater') #alternative='two-sided')
        pvals[i] = p
    pvals = pvals.reshape(*X.shape[1:])
    mask, _ = fdr_correction(pvals, alpha=0.05)
    pvalsm =np.ma.masked_where(~mask, pvals)
    return pvalsm

def plot_gat_scores(data, ax, start=-0.2, stop=1.):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    data_arr = np.array(data) 
    data_av = data_arr.mean((0,1)) # mean sbjs and folds for plotting
    data_arr = data_arr.mean(1) #mean folds
    n_time = len(data_av)
    times = np.linspace(start, stop, n_time)
    X = data_arr - 0.5
    nsbj = len(X) 
    # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(X, threshold=0.01, seed=41, 
    #                                          n_jobs=-1, n_permutations=1024)
    # T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(X, threshold=0.05, seed=41, 
    #                                         n_jobs=-1, n_permutations=1024, out_type='mask')
    # print(cluster_p_values)

    im = ax.imshow(data_av, interpolation='lanczos', origin='lower', cmap='RdBu_r', #,vmin=0.4, vmax=0.6)
                    extent=times[[0, -1, 0, -1]], vmin=0.4, vmax=0.6)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    # for i, p  in enumerate(cluster_p_values):
    #     if p < 0.01:
    #         print(p)
    #         ax.contour(clusters[i], colors='k', origin='lower', extent=times[[0, -1, 0, -1]])

    mask = mass_univ_corrected(X)    
    ax.contour(mask, colors='k', origin='lower', extent=times[[0, -1, 0, -1]], linewidths=3, linestyles='dotted')     

    return 
    
store1, store2, store3, store4, store5 = [[] for i in range(5)]

for sbj in sbjs:
    if sbj == 'VP12':
        continue
    scores1, scores2, scores3, scores4, scores5 = pickle_load(os.path.join(save_path, sbj + '.p'))
    store1.append(scores1)
    store2.append(scores2)
    store3.append(scores3)
    store4.append(scores4)
    store5.append(scores5)
  
fig, ax = plt.subplots(1,5, figsize=(19,8))
times = np.linspace(-0.2, 1., 75)

if 'voices' in save_path:
    xlabels = [ 'familiarN1 vs unfamiliarN1', 'familiarN2 vs unfamiliarN2', 'familiarN3 vs unfamiliarN3',
                'familiarREM vs unfamiliarREM','familiarW vs unfamiliarW']
else:
    xlabels = ['ownN1 vs unknownN1','ownN2 vs unknownN2', 'ownN3 vs unknownN3','ownREM vs unknownREM', 'ownW vs unknownW']

for i, (score, ylabel, xlabel) in enumerate(zip([store1, store2, store3, store4, store5],
                                    [re.sub('/', '', (' vs ').join(contrast1)) for i in range(5)], 
                                    xlabels)):
    print(i)
    if len(score[0])== 0 :
        print('skipped..')
        continue # N1 empty when 50 sampling
    plot_gat_scores(score, ax[i], start=-0.2, stop=1.)
    ax[i].set_ylabel(ylabel)
    ax[i].set_xlabel(xlabel)
plt.tight_layout()

plt.savefig(os.path.join(save_path, re.sub('/', '', ('_').join(contrast1)) + 'mw_2sid.tiff' ), bbox_inches="tight")
plt.show()


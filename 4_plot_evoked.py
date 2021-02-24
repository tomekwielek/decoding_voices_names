import mne
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
from utils import pickle_load
from config import (get_path, myload, mysave, base_path_data, sleep_path_data, 
                    wake_event_id, sleep_event_id)
from mne import grand_average
from mne.viz import plot_epochs_image
from config import mysave
from utils import pickle_save, pickle_load

from mne.stats import spatio_temporal_cluster_test
from mne.stats import permutation_t_test, permutation_cluster_test
import re 

SLEEP = False

save_path = r'D:\SON_deep\results\evokeds\sleep\voices'

if SLEEP:
    path = sleep_path_data
    event_id = sleep_event_id

    # contrasts = [('familiar/N1', 'unfamiliar/N1'),
    #             ('familiar/N2', 'unfamiliar/N2'),
    #             ('familiar/N3', 'unfamiliar/N3'),
    #             ('familiar/REM', 'unfamiliar/REM')]

    # contrasts = [('own/N1', 'un/N1'),
    #         ('own/N2', 'un/N2'),
    #         ('own/N3', 'un/N3'),
    #         ('own/REM', 'un/REM')]            
            
else:
    path = base_path_data
    event_id = wake_event_id

    contrasts = [('own', 'un'),
                ('familiar', 'unfamiliar')]

sbjs = os.listdir(path)
sbjs = [sbjs[i] for i in range(len(sbjs)) if sbjs[i].startswith('VP')]

mylist = []
for sbj in sbjs:
    if sbj == 'VP12':
        continue
    evoked = myload(path, typ='evoked', sbj=sbj, preload=True)
    mylist.append(evoked)

store_dicts = [] 
for contrast in contrasts:
    evoked_dict = dict(zip(contrast, [[] for i in range(len(contrast))]))
    for cond in contrast:
        ga = grand_average( [mylist[i][cond] for i in range(len(mylist)) ])
        ga._data /= 10e5 
        evoked_dict[cond] = ga

    #evoked_dict[cond]  = [evoked_dict[cond]._data / 10e5 for cond in contrast]    
    
    title = (' vs ').join([s.upper() for s in contrast])
    
    mne.viz.plot_compare_evokeds(evoked_dict, combine='median',  title=title, truncate_yaxis=False,
                                ylim=dict(eeg=[-1.4,  1 ]))    

    store_dicts.append(evoked_dict)
    plt.savefig(os.path.join(save_path, re.sub('/', '', ('_').join(contrast))+ '_evokeds.tiff' ), bbox_inches="tight")


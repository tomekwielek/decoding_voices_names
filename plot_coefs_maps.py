import mne
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from utils import pickle_load, pickle_save, my_equalize_events
from config import (get_path, myload, mysave, base_path_data, sleep_path_data, wake_event_id, sleep_event_id)
from IPython.core.debugger import set_trace
from sklearn.pipeline import Pipeline
from mne.decoding import GeneralizingEstimator
from sklearn.linear_model import LogisticRegression
from mne.epochs import equalize_epoch_counts
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import re 
from scipy.stats import sem
from mne.stats import permutation_cluster_1samp_test
from scipy import stats as stats
import matplotlib.patches as mpatches
from mne.decoding import LinearModel, Vectorizer, get_coef
from mne import EvokedArray
import warnings
from sklearn.svm import SVC
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#plt.rcParams.update({'font.size': 16})

#sleep
#contrast1 = ('own/REM', 'un/REM')

#wake
contrast1 = ('familiar/N2', 'unfamiliar/N2')

#contrast1 = ('own', 'un')

save_path = r'D:\SON_deep\results\decod\cross\tmp\voices\coefs'
#save_path = r'D:\SON_deep\results\decod\cross\tmp\voices\coefs'

def get_Xy_balanced(epochs, conditions, n_sample=100):
    epochs1 = epochs[conditions[0]]
    epochs2 = epochs[conditions[1]]
    equalize_epoch_counts([epochs1, epochs2], method='truncate')
    X1 = epochs1._data[:n_sample,...]
    X2 = epochs2._data[:n_sample,...]
    y1 = [0 for i in range(len(X1))] 
    y2 = [1 for i in range(len(X2))] 
    return np.vstack([X1, X2]), np.asarray(y1 + y2)

sbjs = os.listdir(base_path_data)
sbjs = [sbjs[i] for i in range(len(sbjs)) if sbjs[i].startswith('VP')]

tmin, tmax = (0., 0.8)
evokeds = [] 
for sbj in sbjs:
    print(sbj)
    if sbj == 'VP12': #No REM here
        continue
    if os.path.exists(os.path.join(save_path, sbj + '.p')):
        continue
    #sleep_epochs = myload(base_path_data, typ='epoch_preprocessed', sbj=sbj, preload=True) # WAKE!
    sleep_epochs = myload(sleep_path_data, typ='epoch_preprocessed', sbj=sbj, preload=True) # SLEEP!
    sleep_epochs.event_id = sleep_event_id # event_id remapping. For wake this step works during preprocessing # SLEEP !

    sleep_epochs = sleep_epochs.crop(tmin=tmin, tmax=tmax)
    
    X1, y1 = get_Xy_balanced(sleep_epochs, contrast1)
    
    clf =  make_pipeline(Vectorizer(), StandardScaler(), LinearModel(LogisticRegression(max_iter = 4000))) #StandardScaler(),
     
    cv  = StratifiedKFold(n_splits=2, shuffle=True)
    
    coef_folds = [] 
    for train_idx, test_idx in cv.split(X1, y1):
        clf.fit(X1[train_idx], y=y1[train_idx])
        #scores1.append(clf.score(X1[test_idx], y=y1[test_idx]))
        coef_folds.append(get_coef(clf, attr='patterns_', inverse_transform=True))
    coef = np.asarray(coef_folds).mean(0).reshape([173, -1]) #mean folds and reshape
    evoked = EvokedArray(coef, sleep_epochs.info, tmin=tmin)
    evokeds.append(evoked)

ga = mne.grand_average(evokeds)

#SLEEP
f = ga.plot_topomap([0., 0.2, 0.4,  0.6, 0.8], scalings=0.1, vmin=-2, vmax=2)
#f = ga.plot_topomap([0.1, 0.2,  0.3, 0.4, 0.5, 0.6,  0.7, 0.8], scalings=0.1, vmin=-6, vmax=6)
#f = ga.plot_topomap([0.1, 0.15,  0.2, 0.25, 0.3, 0.35,  0.4, 0.45, 0.5], scalings=0.1, vmin=-6, vmax=6)

f.savefig(os.path.join(save_path, contrast1[0].split('/')[1]  + 'coefs_new_detail.tiff'))

#WAKE
# f = ga.plot_topomap([0., 0.2, 0.4,  0.6, 0.8], scalings=0.1, vmin=-2, vmax=2) 
# f.savefig(os.path.join(save_path, 'wakecoefs_new.tiff') )
    
#plt.close()



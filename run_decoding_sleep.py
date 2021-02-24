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
from sklearn.svm import SVC
from mne.epochs import equalize_epoch_counts
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import re 
from scipy.stats import sem
from mne.stats import permutation_cluster_1samp_test
from scipy import stats as stats
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

stages = ['N2']
nsample = 105 # 100=standard, 110=long epochs, 105=standard shuffled

# def get_Xy_balanced(epochs, conditions, n_sample):
#     epochs1 = epochs[conditions[0]]
#     epochs2 = epochs[conditions[1]]
#     equalize_epoch_counts([epochs1, epochs2], method='truncate')
#     if n_sample == None:
#         X1 = epochs1._data
#         X2 = epochs2._data
#     else:       
#         X1 = epochs1._data[:n_sample,...]
#         X2 = epochs2._data[:n_sample,...]
#     y1 = [0 for i in range(len(X1))] 
#     y2 = [1 for i in range(len(X2))] 
#     return np.vstack([X1, X2]), np.asarray(y1 + y2)

def get_Xy_balanced(epochs, conditions, n_sample): #with shuffling
    epochs1 = epochs[conditions[0]]
    epochs2 = epochs[conditions[1]]
    equalize_epoch_counts([epochs1, epochs2], method='truncate')
    if n_sample == None:
        X1 = epochs1._data
        X2 = epochs2._data
    else:       
        X1 = epochs1._data
        X2 = epochs2._data
        np.random.shuffle(X1)
        np.random.shuffle(X2)
        X1 = X1[:n_sample, ...]
        X2 = X2[:n_sample, ...]
    y1 = [0 for i in range(len(X1))] 
    y2 = [1 for i in range(len(X2))] 
    return np.vstack([X1, X2]), np.asarray(y1 + y2)
    
for stage in stages:
    # # #VOICES
    # dname = '{}trained_ordered{}'.format(stage, nsample)
    # save_path = r'D:\SON_deep\results\decod\cross\tmp\voices\{}'.format(dname)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)  

    # if stage == 'N1':
    #     #N1
    #     contrast1 = ('familiar/N1', 'unfamiliar/N1')
    #     contrast2 = ('familiar/N2', 'unfamiliar/N2')
    #     contrast3 = ('familiar/N3', 'unfamiliar/N3')
    #     contrast4 = ('familiar/REM', 'unfamiliar/REM')
    #     contrast5 = ('familiar', 'unfamiliar') #wake
    #     saveorder = [0, 1, 2, 3, 4]
         
    # elif stage == 'REM':
    #     #REM
    #     contrast1 = ('familiar/REM', 'unfamiliar/REM')
    #     contrast2 = ('familiar/N1', 'unfamiliar/N1')
    #     contrast3 = ('familiar/N2', 'unfamiliar/N2')
    #     contrast4 = ('familiar/N3', 'unfamiliar/N3')
    #     contrast5 = ('familiar', 'unfamiliar') 
    #     saveorder = [1, 2, 3, 0, 4]
    
    # elif stage == 'N2':
    #     #N2
    #     contrast1 = ('familiar/N2', 'unfamiliar/N2')
    #     contrast2 = ('familiar/N1', 'unfamiliar/N1')
    #     contrast3 = ('familiar/N3', 'unfamiliar/N3')
    #     contrast4 = ('familiar/REM', 'unfamiliar/REM')
    #     contrast5 = ('familiar', 'unfamiliar') #wake
    #     saveorder = [1, 0, 2, 3, 4]
    # elif stage == 'N3':
    #     #N3
    #     contrast1 = ('familiar/N3', 'unfamiliar/N3')
    #     contrast2 = ('familiar/N1', 'unfamiliar/N1')
    #     contrast3 = ('familiar/N2', 'unfamiliar/N2')
    #     contrast4 = ('familiar/REM', 'unfamiliar/REM')
    #     contrast5 = ('familiar', 'unfamiliar') #wake
    #     saveorder = [1, 2, 0, 3, 4]

    # #NAMES
    dname = '{}trained_ordered{}'.format(stage, nsample)
    save_path = r'D:\SON_deep\results\decod\cross\tmp\names\{}'.format(dname)
    if not os.path.exists(save_path):
        os.makedirs(save_path)  

    if stage == 'N1':
        #N1
        contrast1 = ('own/N1', 'un/N1')
        contrast2 = ('own/N2', 'un/N2')
        contrast3 = ('own/N3', 'un/N3')
        contrast4 = ('own/REM', 'un/REM')
        contrast5 = ('own', 'un') #wake
        saveorder = [0, 1, 2, 3, 4]
         
    elif stage == 'REM':
        #REM
        contrast1 = ('own/REM', 'un/REM')
        contrast2 = ('own/N1', 'un/N1')
        contrast3 = ('own/N2', 'un/N2')
        contrast4 = ('own/N3', 'un/N3')
        contrast5 = ('own', 'un') 
        saveorder = [1, 2, 3, 0, 4]
    
    elif stage == 'N2':
        #N2
        contrast1 = ('own/N2', 'un/N2')
        contrast2 = ('own/N1', 'un/N1')
        contrast3 = ('own/N3', 'un/N3')
        contrast4 = ('own/REM', 'un/REM')
        contrast5 = ('own', 'un') #wake
        saveorder = [1, 0, 2, 3, 4]
    elif stage == 'N3':
        #N3
        contrast1 = ('own/N3', 'un/N3')
        contrast2 = ('own/N1', 'un/N1')
        contrast3 = ('own/N2', 'un/N2')
        contrast4 = ('own/REM', 'un/REM')
        contrast5 = ('own', 'un') #wake
        saveorder = [1, 2, 0, 3, 4]

    sbjs = os.listdir(base_path_data)
    sbjs = [sbjs[i] for i in range(len(sbjs)) if sbjs[i].startswith('VP')]

    for sbj in sbjs:
        print(sbj)
        if sbj == 'VP12': #No REM here
            continue
        # if os.path.exists(os.path.join(save_path, sbj + '.p')):
        #     continue
        wake_epochs = myload(base_path_data, typ='epoch_preprocessed', sbj=sbj, preload=True)
        sleep_epochs = myload(sleep_path_data, typ='epoch_preprocessed', sbj=sbj, preload=True)
        sleep_epochs.event_id = sleep_event_id # event_id remapping. For wake this step works during preprocessing
        wake_epochs.event_id = wake_event_id # only for long

        #X1, y1 = get_Xy_balanced(sleep_epochs, contrast1, n_sample=None)
        X1, y1 = get_Xy_balanced(sleep_epochs, contrast1, n_sample=nsample)
        X2, y2 = get_Xy_balanced(sleep_epochs, contrast2, n_sample=nsample)
        X3, y3 = get_Xy_balanced(sleep_epochs, contrast3, n_sample=nsample)
        X4, y4 = get_Xy_balanced(sleep_epochs, contrast4, n_sample=nsample)
        X5, y5 = get_Xy_balanced(wake_epochs, contrast5, n_sample=nsample)

        del wake_epochs
        del sleep_epochs

        clf = GeneralizingEstimator(make_pipeline(StandardScaler(), LogisticRegression(max_iter = 4000)),
                                    scoring='accuracy', n_jobs=6)
        # clf = GeneralizingEstimator(make_pipeline(StandardScaler(), SVC(kernel='linear')),
        #                             scoring='accuracy', n_jobs=6)
        #clf = GeneralizingEstimator(make_pipeline(StandardScaler(), SVC(kernel='linear')),
                                   

        cv  = StratifiedKFold(n_splits=2, shuffle=True)

        scores1, scores2, scores3, scores4, scores5 = [ [] for i in range(5) ]

        for train_idx, test_idx in cv.split(X1, y1):
            clf.fit(X1[train_idx], y=y1[train_idx])
            scores1.append(clf.score(X1[test_idx], y=y1[test_idx]))
        scores2.append(clf.score(X2, y=y2))
        scores3.append(clf.score(X3, y=y3))
        scores4.append(clf.score(X4, y=y4))
        scores5.append(clf.score(X5, y=y5))
            
        results = [scores1, scores2, scores3, scores4, scores5]
        results = [results[i] for i in saveorder]
        pickle_save(os.path.join(save_path, sbj + '.p'), results)
        




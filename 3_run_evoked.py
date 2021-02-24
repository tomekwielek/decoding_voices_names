import mne
import numpy as np
import os
from IPython.core.debugger import set_trace
from config import (get_path, myload, mysave, base_path_data, sleep_path_data, wake_event_id, sleep_event_id)

SLEEP = True

if SLEEP:
    path = sleep_path_data
    event_id = sleep_event_id
    conditions = ['own/N1', 'own/N2', 'own/N3', 'own/REM',
                'un/N1', 'un/N2', 'un/N3', 'un/REM',
                'familiar/N1', 'familiar/N2', 'familiar/N3', 'familiar/REM',
                'unfamiliar/N1', 'unfamiliar/N2', 'unfamiliar/N3', 'unfamiliar/REM']
else:
    path = base_path_data
    event_id = wake_event_id
    conditions = ['own',
                'un', 
                'familiar', 
                'unfamiliar']

sbjs = os.listdir(path)
sbjs = [sbjs[i] for i in range(len(sbjs)) if sbjs[i].startswith('VP')]
mydict = dict(zip(conditions, [[] for i in range(len(conditions))]))

for sbj in sbjs:
    epochs = myload(path, typ='epoch', sbj=sbj, preload=True)
    epochs.event_id = event_id
    epochs_filt = epochs.filter(1, 20)
    epochs_filt = epochs_filt.crop(-0.2, 1.)
    for c in conditions:
        mydict[c] = epochs_filt[c].average()
    mysave(path, mydict, typ='evoked', sbj=sbj, overwrite=True)



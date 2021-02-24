import mne
import numpy as np
import os
import pandas as pd
from utils import pickle_load, pickle_save
from config import (get_path, myload, mysave, base_path_data, sleep_path_data, wake_event_id, sleep_event_id)
from IPython.core.debugger import set_trace

SLEEP = False

if SLEEP:
    path = sleep_path_data
    event_id = sleep_event_id
else:
    path = base_path_data
    event_id = wake_event_id

sbjs = os.listdir(path)
sbjs = [sbjs[i] for i in range(len(sbjs)) if sbjs[i].startswith('VP')]

for sbj in sbjs: 
    epochs = myload(path, typ='epoch', sbj=sbj, preload=True)
    #epochs.event_id = event_id # event_id remapping doesn't work for sleep data at this stage. Remap later on.
    epochs = epochs.filter(1, 20)
    #epochs.crop(-0.2, 1.)
    epochs.crop(-0.2, 1.5)
    epochs = epochs.decimate(decim=4)

    #mysave(path, epochs, typ='epoch_preprocessed', sbj=sbj, overwrite=True)
    mysave(path, epochs, typ='epoch_preprocessed_long', sbj=sbj, overwrite=True)

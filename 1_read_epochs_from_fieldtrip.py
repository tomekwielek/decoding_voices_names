from config import raw_path_sleep, raw_path, mysave, base_path_data, sleep_path_data
import os
import numpy as np
import scipy.io as sio
import mne
from mne import create_info
from IPython.core.debugger import set_trace
import h5py

PATH = raw_path
BASE_PATH = base_path_data

def info_from_mat(path):
    # load first file to get info (assum: all files have same structure e.g. chs count, sfreq etc)
    first_file = sorted(os.listdir(path))[0] 
    file_path = os.path.join(path, first_file)
    if 'sleep' in PATH: 
        with h5py.File(file_path) as f:
            sfreq = f['data_reref/fsample'][:]#get sampling freq
            sfreq = sfreq[0][0]
            ch_names = []
            for trial in f['data_reref/label']:
                ch_names.append(''.join(map(chr, f[trial[0]][:])))
        f.close()
    else:
        mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        ft_data = mat['data_reref']
        ch_names = list(ft_data.label)
        sfreq = float(ft_data.fsample) 
    montage = 'GSN-HydroCel-257' 
    info = create_info(ch_names=ch_names, sfreq=sfreq,ch_types='eeg',montage=montage)
    return info

def comine_sessions(file_list):
    tmp = []
    for f in file_list:
        full_path = os.path.join(PATH, f)
        epoch = mne.io.read_epochs_fieldtrip(full_path, info=info, data_name='data_reref', trialinfo_column=0)
        tmp.append(epoch)   
    return mne.concatenate_epochs(tmp)

info = info_from_mat(path=PATH)
mat_files = os.listdir(PATH)
mat_files = [f for f in mat_files if f.endswith('.mat')]


if 'sleep' in PATH: 
    #process sleep: single session
    for f in mat_files:
        full_path = os.path.join(PATH, f)
        epoch = mne.io.read_epochs_fieldtrip(full_path, info=info, data_name='data_reref', trialinfo_column=0)
        mysave(BASE_PATH, epoch, typ='epoch', sbj=f.split('_')[0], overwrite=True)
else:
    #process raw: 4 sessions to be combined
    mat_files_idx = [f.split('_')[0] for f in mat_files]
    unique_idx = np.unique(mat_files_idx)
    for sbj in unique_idx:
        bool_mask = [m.startswith(sbj) for m in mat_files_idx]
        files_to_combine = np.asarray(mat_files)[bool_mask]
        print(files_to_combine)
        epoch = comine_sessions(files_to_combine)
        mysave(BASE_PATH, epoch, typ='epoch', sbj=sbj, overwrite=True)



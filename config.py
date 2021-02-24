import os.path as op
import os
import pickle
import numpy as np
from collections import OrderedDict

from mne import read_epochs, read_evokeds
from mne.io import read_raw_fif


if os.path.isdir('D:\\SON_deep\\data\\'): #windows
    base_path_data = 'D:\\SON_deep\\data\\'
    sleep_path_data = 'D:\\SON_deep\\data_sleep\\'
    raw_path = 'D:\\MNE\\moham_analysis\\data\\wake\\raws\\'
    raw_path_sleep = 'D:\\MNE\\moham_analysis\\data\\sleep\\raws\\'
elif os.path.isdir( '/home/b1016533/SON_deep/'): #Linux
    base_path_data = '/home/b1016533/SON_deep/data/'
    sleep_path_data = '/home/b1016533/SON_deep/data_sleep/'



def get_path(base_path, typ, sbj=None, c=None):
    sbj = 'VP'+ str('%02d') % sbj if isinstance(sbj, int) else sbj
    this_path = op.join(base_path, sbj, typ)
    path_template = dict(
        epoch = op.join(this_path, '%s-epo.fif' % (sbj)),
        evoked = op.join(this_path, '%s-ave.fif' % (sbj)),
        epoch_preprocessed = op.join(this_path, '%s-epo.fif' % (sbj)),
        epoch_preprocessed_long = op.join(this_path, '%s-epo.fif' % (sbj)),
        gat_score = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        gat_score_thetainduced = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        gat_score_deltainduced = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        gat_score_alphainduced = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        gat_score_evoked_alphainduced = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        gat_score_evoked_thetainduced = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        gat_score_evoked_deltainduced = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        gat_score_delta = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        t_score = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        t_score_csp = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        freq_scores = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        xdawn = op.join(this_path, '%s%s.pickle' % (sbj, c)),
        tfr_full = op.join(this_path, '%s.pickle' % (sbj)),
        tfr_delta = op.join(this_path, '%s.pickle' % (sbj)),
        epoch_env_delta = op.join(this_path, '%s-epo.fif' % (sbj)),
        epoch_env_theta = op.join(this_path, '%s-epo.fif' % (sbj)),
        epoch_env_theta_induced = op.join(this_path, '%s-epo.fif' % (sbj)),
        epoch_env_delta_induced = op.join(this_path, '%s-epo.fif' % (sbj)),
        epoch_env_alpha_induced = op.join(this_path, '%s-epo.fif' % (sbj)),
        epoch_env_beta_induced = op.join(this_path, '%s-epo.fif' % (sbj)),
        epoch_env_alpha = op.join(this_path, '%s-epo.fif' % (sbj))
        )

    this_file = path_template[typ]
    # Create subfolder if necessary
    folder = os.path.dirname(this_file)
    if (folder != '') and (not op.exists(folder)):
        os.makedirs(folder)
    return this_file

def myload(base_path, typ, sbj=None, c=None, preload=False):
    fname = get_path(base_path=base_path, typ=typ, sbj=sbj, c=c)
    if typ in ['epoch', 'epoch_preprocessed', 'epoch_env_delta', 'epoch_env_theta', 'epoch_env_alpha', 'epoch_env_theta_induced',
                'epoch_env_delta_induced', 'epoch_env_alpha_induced', 'epoch_env_beta_induced', 'epoch_preprocessed_long']:
        out = read_epochs(fname, preload=preload) 
    elif typ in ['evoked', 'gat_score', 't_score', 't_score_csp', 'freq_scores', 'xdawn', 'tfr_full','tfr_delta', 
                'gat_score_delta', 'gat_score_thetainduced', 'gat_score_deltainduced', 'gat_score_alphainduced',
                'gat_score_evoked_alphainduced', 'gat_score_evoked_thetainduced', 'gat_score_evoked_deltainduced']:
        with open(fname, 'rb') as f:
            out = pickle.load(f,  encoding='latin1')
    elif typ in ['raw']:
        out = read_raw_fif(fname, preload=preload)
    else:
        raise NotImplementedError()
    return out

def mysave(base_path, var, typ, sbj, c=None, overwrite=True):
    # get file name
    fname = get_path(base_path, typ, sbj=sbj, c=c)
    # check if file exists
    if op.exists(fname) and not overwrite:
        print('%s already exists. Skipped' % fname)
        return False
    # different data format depending file type
    if typ in ['raw']:
        var.save(fname, overwrite=overwrite)
    elif typ in ['epoch', 'epoch_preprocessed',  'epoch_env_delta',  'epoch_env_theta', 'epoch_env_alpha', 'epoch_env_theta_induced',
                'epoch_env_delta_induced', 'epoch_env_alpha_induced', 'epoch_env_beta_induced', 'epoch_preprocessed_long']:
        var.save(fname)
    elif typ in ['evoked', 'gat_score', 't_score', 't_score_csp', 'freq_scores', 'xdawn', 'tfr_full', 'tfr_delta',
                'gat_score_delta', 'gat_score_thetainduced', 'gat_score_deltainduced', 'gat_score_alphainduced',
                'gat_score_evoked_alphainduced', 'gat_score_evoked_thetainduced', 'gat_score_evoked_deltainduced']:
        with open(fname, 'wb') as f:
            pickle.dump(var, f, protocol=4)
        print('Saving {}'.format(fname))
    else:
        raise NotImplementedError()
    return False

wake_event_id = {'familiar/own' : 12, 'familiar/un/un1' :23, 'familiar/un/un2' : 34, \
        'unfamiliar/own' : 11, 'unfamiliar/un/un1' :22, 'unfamiliar/un/un2' :33}
sleep_event_id = {
    'familiar/own/N1': 21,
    'familiar/own/N2': 22,
    'familiar/own/N3': 23,
    'unfamiliar/own/N1': 11,
    'unfamiliar/own/N2': 12,
    'unfamiliar/own/N3': 13,
    'familiar/un/N1/a': 41,
    'familiar/un/N1/b': 61,
    'familiar/un/N2/a': 42,
    'familiar/un/N2/b': 62,
    'familiar/un/N3/a': 43,
    'familiar/un/N3/b': 63,
    'unfamiliar/un/N1/a': 31,
    'unfamiliar/un/N1/b': 51,
    'unfamiliar/un/N2/a': 32,
    'unfamiliar/un/N2/b': 52,
    'unfamiliar/un/N3/a': 33,
    'unfamiliar/un/N3/b': 53,
    'familiar/own/REM': 25,
    'unfamiliar/own/REM': 15,
    'familiar/un/REM/a': 45,
    'familiar/un/REM/b': 65,
    'unfamiliar/un/REM/a': 35,
    'familiar/un/REM/b': 55
    }     

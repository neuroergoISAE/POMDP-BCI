"""
Utils for experiments

Author: Juan Jesús Torre
Mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import mne
import os
import warnings

import numpy as np
import pandas as pd

from moabb.datasets import Nakanishi2015
from scipy.stats import pearsonr


def get_liveamp_montage(montage_path):
    """
    Gets a standard liveamp montage for MNE

    Parameters
    ----------

    montage_path: str or path object
        Absolute path to the montage file

    Returns
    -------

    montage: mne DigMontage object
        Montage object to pass to the raw data
    """

    montage = pd.read_csv(os.path.join(montage_path, 'live_amp.ced'), sep='\t')
    montage.columns = montage.columns.str.strip()
    montage['names'] = range(1, 33)
    montage.names = montage.names.apply(str)
    montage["names"] = montage["names"].str.strip()
    montage = montage.set_index('names')
    montage = montage.to_dict(orient="index")
    for channel in montage.keys():
        yxz = np.array([-1 * montage[channel]["Y"], montage[channel]["X"], montage[channel]["Z"]])
        montage[channel] = yxz
    montage = mne.channels.make_dig_montage(montage, coord_frame='head')

    return montage


def load_data(subject, dataset, eeg_path=None, ch_keep=[]):
    """
    Get the MNE raw file for a particular subject and dataset

    Parameters
    ----------

    subject: int
        Subject number

    dataset: str
        Name of the dataset to load from

    eeg_path: str or None, default None
        Path to load data from, in case it is handled directly

    ch_keep: list, default=[]
        List containing the channel names that need to be kept

    Returns
    -------

    Epochs: mne.Epochs object
        Preprocessed and epoched data
    """

    if dataset == 'nakanishi':
        nakanishi = Nakanishi2015()
        sessions = nakanishi.get_data(subjects=[subject])
        file_path = nakanishi.data_path(subject)

        raw = sessions[subject]['session_0']['run_0']
        events = mne.find_events(raw, verbose=False)
        event_id = nakanishi.event_id

    else:
        if dataset == 'high_amp':
            filename = f"P{subject}_low_100.set"
        elif dataset == 'cvep':
            filename = f"{subject}_mseqwhite.set"

        file_path = os.path.join(eeg_path, filename)
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

        # CVEP needs annotation cleaning before extracting events
        if dataset == 'cvep':
            for idx in range(len(raw.annotations.description)):
                code = raw.annotations.description[idx].split('_')[0]
                lab = raw.annotations.description[idx].split('_')[1]
                code = code.replace('\n', '')
                code = code.replace('[', '')
                code = code.replace(']', '')
                code = code.replace(' ', '')
                raw.annotations.description[idx] = code + '_' + lab

        events, event_id = mne.events_from_annotations(raw, verbose=False)

    print('')
    print('-' * 70)
    print(f"File path: {file_path}")
    print('-' * 70)

    print('')
    print(f'Loaded data for subject {subject}')

    # Preprocessing
    raw = raw.drop_channels(list(set(raw.ch_names) - set(ch_keep)))

    # CVEP needs the montage manually set
    if dataset == 'cvep':
        montage = get_liveamp_montage(eeg_path)
        raw.set_montage(montage)
        raw = raw.drop_channels(['21', '10'])

    mne.set_eeg_reference(raw, 'average', copy=False, verbose=False)
    print("Dropped unnecesary EEG channels")
    print(f"Channels kept: {raw.ch_names}")

    raw = raw.filter(l_freq=3, h_freq=90, method="iir", verbose=False)  # Maybe unnecessary due to filterbank
    print("Data was filtered")

    # Epoching
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=0, tmax=2.2, baseline=(0.2, 2.2),
                        preload=True, verbose=False)
    print("Data was epoched")

    return epochs

def add_safety_margin(conf_matrix, mixing_coef=0.3):
    """
    Modify the confusion matrix by mixing it with the uniform distribution, according to what is 
    proposed in [1]
   
    Parameters
    ----------

    conf_matrix: np.array of shape (n_classes, n_classes)
        Confusion matrix to modify

    mixing_coeff: float, default=0.3
        Parameter that determines the maximum value of the diagonal after the operation, that will
        be roughly equal to 1 - mixing_coeff

    References:

        [1] - Park, J., & Kim, K. E. (2012).
              A POMDP approach to optimizing P300 speller BCI paradigm.
              IEEE Transactions on Neural Systems and Rehabilitation Engineering, 20(4), 584-594.
    """

    copy_matrix = conf_matrix.copy()

    n_class = copy_matrix.shape[0]
    copy_matrix = (1 - mixing_coef) * copy_matrix + mixing_coef * 1 / n_class

    return copy_matrix


def epoch_to_window(data, labels, sfreq, codes, win_len=0.250, win_format='channels_last', refresh=60,
                    return_code_len=True):
    """
    Function to transform EEG epochs (mne format) into CNN-compatible data windows.

    Parameters
    ----------

    data: np.array, shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
        EEG data as loaded by MNE

    labels: np.array, shape (n_trials,) or int
        EEG labels corresponding to the data

    sfreq: int
        Sampling frequency of the data

    codes: OrderedDict
        OrderedDict containing the labels as keys and the codes as values for every class

    win_len: float, default=0.250
        Length of the EEG window. Default value taken from the original EEG2Code paper

    win_format: str, ['channels_first', 'channels_last'], default='channels_first
        Format of the windowd data. A new dimension is inserted at the end of the process, its position
        depending on the selected value of the parameter. If an invalid value is passed, 'channels_last'
        will be used

    refresh: int, default=60
        Refresh rate of the monitor used to present the codes. Change accordingly to correctly match code
        changes with the data

    return_code_len: bool, default True
        If True, return the length of the code (per trial) in samples as well as the windowed data and labels.
        Used for testing

    Returns
    -------

    win_data: np.array, shape (n_samples, n_channels, width, height) or (n_samples, width, height, n_channels)
        Windowed data. The EEG epoch is treated as a gray-scale image (1 channel) of width equal to n_channels
        and height equal to epoch_len, with each sample being a bit of the code

    win_labels: np.array, shape (n_samples,)
        Windowed data. Instead of a label per trial, the new labels array contains a label per sample, corresponding
        to wich bit of code matches the corresponding data sample. Since the sampling frequency of EEG differs from
        the refresh rate of the monitor (frequency at which the code can change), a number of consecutive data samples
        will have the same label, that changes with the screen refresh rate

    code_len: int
        Length of the code (per trial)
    """
    # Handle single trial by inserting new dim at axis 0
    if data.ndim == 2:
        data = np.expand_dims(data, 0)

    # Get useful parameters from the data
    n_trials, n_channels, n_samples = data.shape
    win_samples = int(win_len * sfreq)
    code_len = int(n_samples - win_samples)  # int((2.2-0.250)*sfreq)

    # Initialize final arrays
    win_data = np.empty(shape=(code_len * n_trials, n_channels, win_samples))
    win_labels = np.empty(shape=(code_len * n_trials), dtype=int)

    # Convert to window
    sample_count = 0  # Track the current sample in the data
    for trial_n, trial in enumerate(data):
        # Get next trial and label
        try:  # labels is an int if only one trial is passed
            label = labels[trial_n]
        except TypeError:
            label = labels
        code = codes[label]

        # Iterate over samples, keeping track of the corresponding code bit
        code_pos = 0  # Track the frames of the screen / code bits
        for idx in range(code_len):
            win_data[sample_count] = trial[:, idx:idx + win_samples]
            # If the current sample is over the duration of the current bit, increase the bit count
            if idx/sfreq >= (code_pos+1) / refresh:
                code_pos += 1
            # Assign label to the sample according to the corresponding code bit
            win_labels[sample_count] = int(code[code_pos])
            sample_count += 1

    # Insert new dimension to the data
    if win_format == 'channels_first':
        dim_pos = 1
    elif win_format == 'channels_last':
        dim_pos = -1
    else:
        warnings.warn("Invalid win_format parameter, using 'channels_last as default...", UserWarning)
        dim_pos = -1

    win_data = np.expand_dims(win_data, dim_pos)
    win_data = win_data.astype(np.float32)  # Format as float

    # One-hot encoding of labels for the NN
    win_labels = np.vstack((win_labels, np.abs(1-win_labels))).T

    if return_code_len:
        return win_data, win_labels, code_len
    else:
        return win_data, win_labels


def get_code_prediction(codes_true, codes_pred, sfreq, code_len, t_min=0., t_max=None, refresh=60, incremental=False):
    """
    Make predictions based on regressed codes

    Parameters
    ----------

    codes_true: OrderedDict object
        Dictionary containing the true codes for each class, with the label as the key and the code as value

    codes_pred: np.array object, shape(n_samples,)
        Code regressed from the NN. Each value corresponds to the designated label of an EEG sample. These
        will be averaged to adjust to the refresh rate of the screen before being compared with the true codes.
        Even so the DNN returns an array of shape (n_samples, 2), only one column is necessary for the pre-
        diction. The first one is used for convenience.

    sfreq: int
        Sampling frequency of the EEG data

    code_len: int
        Length of the code per trial. Necessary to correctly separate the continuous code into trials

    t_min: int, default=0.
        Time for the first sample of each trial of the data to be sliced from, in seconds

    t_max: float or None, default=None
        Time for the last sample of each trial of the data to be sliced from, in seconds

    refresh: int, default=60
        Refresh rate of the monitor used to present the codes. Change accordingly to correctly match code
        changes with the data

    incremental: bool, default=False
        Incremental mode switch. If True, t_max is used as the initial t_max of the slice, and then the
        correlation is repeated increasing the t_max in 3 samples until a significant correlation is found
        or the epoch is exhausted.

    Returns
    -------

    labels_pred: np.array, shape (n_trials,)
        List of predicted class labels

    total_mean_time: float
        Average time (across trials) to get a prediction
    """

    # Window len in ms: int((epoch_len - EEG2Code window_len) * sfreq)
    codes_pred = np.array(codes_pred)
    n_trials = int(len(codes_pred)/code_len)

    # Final labels list
    labels_pred = []
    mean_time = []

    for trial in range(n_trials):
        # Get the next trial
        trial_code = codes_pred[trial * code_len:(trial + 1) * code_len]

        # Transform the code from EEG sampling rate to screen refresh rate by averaging predictions
        code_buffer = []
        code_pos = 0
        y_tmp = []
        for idx in range(len(trial_code)):
            y_tmp.append(trial_code[idx])
            if idx/sfreq >= (code_pos + 1) / refresh:
                code_pred = np.mean(y_tmp)
                code_pred = int(np.rint(code_pred))
                code_buffer.append(code_pred)
                y_tmp = []
                code_pos += 1

        # Find the code that correlate the most
        corr = -2
        pred_lab = -1
        out = 0
        sample_start = int(t_min * refresh)
        if not t_max:
            sample_end = len(code_buffer)
        else:
            sample_end = int(t_max * refresh)

        # Temporal shit to keep compatibility with incremental trial len later
        if not incremental:
            points = [(sample_start, sample_end)]
        else:
            max_sample = len(code_buffer)
            points = [(sample_start, end) for end in range(sample_end, max_sample + 1, 3)]

        for start, end in points:
            corrs = []
            # pred_lab = -1
            corr = -2
            tmp = -1
            for key, values in codes_true.items():
                corp, p_value = pearsonr(code_buffer[start:end], values[start:end])
                corrs.append(corp)
                if corp > corr:
                    corr = corp
                    p_max = p_value
                    tmp = key
            # Assign the prediction to the highest correlation
            if not incremental:
                pred_lab = tmp
                mean_time.append(end / refresh)
            else:
                corrs_idx = np.argsort(corrs)
                if ((corrs[corrs_idx[-1]] - corrs[corrs_idx[-2]])/corrs[corrs_idx[-1]] > 0.5) and (p_max < 1e-3):
                    mean_time.append(end / refresh)
                    pred_lab = tmp
                    break
        # Append to the list
        labels_pred.append(pred_lab)

    # Make prediction into array for compatibility
    labels_pred = np.array(labels_pred)
    total_mean_time = sum(mean_time) / len(mean_time)

    if incremental:
        return labels_pred, total_mean_time
    else:
        return labels_pred


def make_preds_accumul_aggresive(codes_true, codes_pred, sfreq, code_len, min_len=30, refresh=60):
    """
    Always make a prediction and it's the template that correlate the most
    during the last step of window growth.
    Stop the process if the same prediction is made 40 times in the row
    and pick this prediction.
    """

    codes_pred = np.array(codes_pred)
    n_trials = int(len(codes_pred)/code_len)

    # Final lists
    labels_pred = []
    mean_time = []

    for trial in range(n_trials):
        # Retrieve a trial
        trial_code = codes_pred[trial * code_len:(trial + 1) * code_len]
        code_pos = 0

        # Do an average over the prdata, codes, labels, sfreq
        code_buffer = []
        y_tmp = []
        for idx in range(len(trial_code)):
            y_tmp.append(trial_code[idx])
            if idx/sfreq >= (code_pos+1) / refresh:
                code_pred = np.mean(y_tmp)
                code_pred = int(np.rint(code_pred))
                code_buffer.append(code_pred)
                y_tmp = []
                code_pos += 1

        # Find the code that correlate the most
        pred_lab = -1
        temp_pred = -1
        out = 0
        for long in np.arange(min_len, len(code_buffer) - 1, step=1):
            dtw_values = []
            for _, values in codes_true.items():
                dtw_values.append(np.corrcoef(code_buffer[:long], values[:long])[0, 1])
            dtw_values = np.array(dtw_values)
            max_dtw = list(codes_true.keys())[np.argmax(dtw_values)]
            if max_dtw == temp_pred:
                out += 1
            else:
                temp_pred = max_dtw
                out = 0
            if out == 15:
                pred_lab = temp_pred
                mean_time.append(long / refresh)
                break
        labels_pred.append(pred_lab)

    labels_pred = np.array(labels_pred)
    total_mean_time = sum(mean_time) / len(mean_time)
    return labels_pred, total_mean_time


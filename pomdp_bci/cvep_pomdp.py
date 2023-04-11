"""
CVEP-POMDP implementation

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import itertools
import json
import keras
import os
import pomdp_py
import random

import numpy as np
import pandas as pd

from collections import OrderedDict, defaultdict
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from pomdp_py import sarsop

from pomdp_bci.problem import BCIProblem
from pomdp_bci.domain import BCIState, BCIObservation
from pomdp_bci.utils import EEGnet_patchembeddingdilation
from pomdp_bci.utils import load_data, add_safety_margin, save_results, epoch_to_window, \
                            get_code_prediction, make_preds_accumul_aggresive


def fit_clf(win_data, win_labels):
    """
    Return a fit classifier using the selected data and architecture

    Parameters
    ----------

    win_data: np.array, shape (n_samples, n_channels, width, height) or (n_samples, width, height, n_channels)
        Windowed data for training

    win_labels: np.array, shape (n_samples,)
        Windowed labels for training

    algorithm: str
        Architecture used for the CNN

    Returns
    -------

    fit_clf: clf that has been fit with the given data and architecture
    """
    # Class balancing: There are more 1s than 0s in our training codes, so we use a random
    # under sampler to make it balanced
    rus = RandomUnderSampler()
    counter = np.array(range(0, len(win_labels))).reshape(-1, 1)
    index, _ = rus.fit_resample(counter, win_labels[:, 0])
    win_data = np.squeeze(win_data[index, :, :, :], axis=1)
    win_labels = np.squeeze(win_labels[index])

    # Initialize NN
    win_samples = int(code_win_len * params['sfreq'])
    n_channels = win_data.shape[1]  # Number of channels in the data (for channels last)

    if algo == 'EEGnet_patchembeddingdilation':
        clf = EEGnet_patchembeddingdilation(windows_size=win_samples, n_channel_input=n_channels)
    else:
        raise NameError("Incorrect algo name")

    print('')
    clf.summary()

    # Select optimizer and compile classifier for fit
    batchsize = 256
    epochs = 45
    learning_rate = 1e-3
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
    clf.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print('')
    print('Classifier compiled. Fitting...')
    print('')
    history = clf.fit(win_data, win_labels, batch_size=batchsize, epochs=epochs, shuffle=True, verbose=0)
    keras.backend.clear_session()

    return clf


# Path variables - put your paths here or export them to your environment
data_path = os.environ.get('CVEP')
sarsop_path = os.environ.get('SARSOP')

time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
results_path = os.path.join(os.getcwd(), f'results/CVEP_{time}')
if not os.path.exists(results_path):
    os.mkdir(results_path)

# Data variables
init_delay = 0.  # In seconds, 0.13 in the original TRCA paper
n_fbands = 5  # Parameter for filterbank
code_win_len = 0.250  # In seconds, 0.250 in the original EEG2Code paper
epoch_len = 1 + 0.01  # Adding 0.01 to get 60 bits at the end
cv_n = 5  # Number of trials (per class) used for Confusion matrix calculation
test_n = 2  # Number of trials (per class) used for POMDP simulation
add_margins = True  # Minimize the impact of zeroes and big numbers on the conf matrix
mixing_coef = 0.3

# POMDP variables
hit_reward = 10
miss_costs = [-100, -1000]
wait_cost = -1

# Since we know the number of steps for each POMDP trial, we can set the gamma automatically instead of heuristically,
# as the value we want for it to be when elevated to the power of the last time step. Knowing that the value formula
# for the POMDP weights future values as gamme ** t, we can specify the value we want gamma to have at t_max
gamma_horizon = 0.25

# Analysis variables
slice_steps = [(0.5, 0.1)]  # (slice_len, time_step)
datasets = ['cvep']
algos = ['EEGnet_patchembeddingdilation']
clf_methods = ['cvep_pvalue', 'cvep_cumul', 'cvep_static']

# Big results filename
grand_results_filename = os.path.join(results_path, f'comparison_cvep_{time}.csv')
grand_results = {}  # Keys are iterations, hold results for all subjects

for dataset, algo in itertools.product(datasets, algos):
    # Load dataset-specific parameters
    with open(f'{config/dataset}.json', 'r') as dataset_params:
        params = json.loads(dataset_params.read())

    downsample = int(params['sfreq'] / 250)  # After downsampling, sfreq should be 250Hz

    sub_list = params['sub_list']
    score_dict = {sub: {} for sub in sub_list}
    metadata_dict = {}

    for sub in sub_list:
        # Load the data (preprocessed and epoched)
        epochs = load_data(subject=sub, dataset=dataset, eeg_path=data_path, ch_keep=params["ch_slice"])

        labels = epochs.events[..., -1]
        labels -= np.min(labels)

        t_min = int(init_delay * params['sfreq']) + 1  # In samples
        t_max = int(t_min + (epoch_len + code_win_len) * params['sfreq'])  # Also in samples

        data = epochs.get_data()
        data = data[..., t_min:t_max]
        print(f"Selecting data from epoch {t_min} to {t_max}, "
              f"corresponding to {init_delay} and {init_delay + epoch_len + code_win_len}s")

        # Analysis
        codes = OrderedDict()  # Make an ordered dict with the code for each class
        for k, v in epochs.event_id.items():
            code = k.split('_')[0]
            idx = k.split('_')[1]
            codes[v - 1] = np.array(list(map(int, code)))

        label_id, label_count = np.unique(labels, return_counts=True)
        n_class = len(label_id)

        print('')
        print(f"Using labels {label_id} with number of trials {label_count}")

        # Divide data in train and test (for baseline), and further divide train data into calibration and cv
        # (for POMDP training and conf matrix creation)
        cal_n = np.unique(label_count)[0] - (cv_n + test_n)  # Number of trials (per class) used for calibration
        train_n = cal_n + cv_n  # Number of trials (per class) used for calibration + conf_matrix generation

        X_train, y_train = data[:train_n * n_class], labels[:train_n * n_class]  # The first train_n trials
        X_test, y_test = data[train_n * n_class:], labels[train_n * n_class:]

        # X_cal is used to train the NN that the POMDP uses, and X_cv to calculate the confusion matrix
        X_cal, y_cal = X_train[:cal_n * n_class], y_train[:cal_n * n_class]
        X_cv, y_cv = X_train[cal_n * n_class:], y_train[cal_n * n_class:]

        # POMDP Model fit
        print('')
        print('Fitting data for POMDP...')
        # We use 'data' and 'labels' for windowed data instead of 'X' and 'y'
        data_cal, labels_cal, code_len = epoch_to_window(X_cal, y_cal, sfreq=params['sfreq'], codes=codes,
                                                         win_len=code_win_len, win_format='channels_last',
                                                         return_code_len=True)
        data_cv, labels_cv, _ = epoch_to_window(X_cv, y_cv, sfreq=params['sfreq'], codes=codes,
                                                win_len=code_win_len, win_format='channels_last',
                                                return_code_len=True)

        # Data needs to be normalized
        cal_std = data_cal.std(axis=0)
        data_cal /= cal_std + 1e-8
        data_cv /= cal_std + 1e-8

        # Fit
        pomdp_clf = fit_clf(data_cal, labels_cal, algorithm=algo)
        print('Data was fit')

        # Predict the codes of validation data (split in 10 to avoid OOM from the GPU)
        pomdp_codes_pred = []
        bits_per_class = int(len(data_cv) / n_class)
        pred_splits = 11
        for i in range(pred_splits):
            if i == pred_splits - 1:
                data_slice = data_cv[bits_per_class * i:]
            else:
                data_slice = data_cv[bits_per_class * i: bits_per_class * (i + 1)]

            pred = pomdp_clf.predict(data_slice, batch_size=64)
            pomdp_codes_pred.extend(pred[:, 0])

        pomdp_codes_pred = np.array(pomdp_codes_pred)  # In window format

        # Once the model is trained and the codes are regressed, create a POMDP for different values of epoch len,
        # time step, miss cost, gamma, etc.
        for (slice_len, time_step), miss_cost in itertools.product(slice_steps, miss_costs):
            # Get the number of steps the model can use to simulate the policy: 6 for 0.1
            model_steps = int((epoch_len - slice_len) / time_step) + 1
            # Get the value that, when to the power of model_steps, gives the desired gamma_horizon
            gamma = np.power(gamma_horizon, (1 / model_steps))

            print('')
            print('Creating confusion matrix, POMDP parameters:')
            print(f'  Window length: {slice_len}')
            print(f'  Time step: {time_step}')
            print(f'  Cost for misses: {miss_cost}')
            print(f'  Discount factor: {gamma}')

            # Create dictionary entry for this iteration
            iter_name = f'{dataset}_{algo}-slice{slice_len}_step{time_step}_cost{miss_cost}_gamma{gamma}'
            score_dict[sub][iter_name] = {}

            # Get partial acc score on every time step
            time_steps = list(np.arange(0, epoch_len + 0.01, time_step))
            total_steps = len(time_steps)
            step_interval = int(slice_len / time_step)
            y_pred = []
            y_long = []

            # Use a list of all time steps from 0 and take the slice by skipping n steps in
            # the list according to step_interval. Stop when your time_end index is out of the list
            print('')
            for step_n in range(total_steps):
                t_start = time_steps[step_n]
                try:
                    t_end = time_steps[step_n + step_interval]
                    y_long.extend(list(y_cv))  # Labels are the same no matter the slice
                except IndexError:
                    continue

                # Predict data on current time step and add to pred array. This goes back to non-window format
                pred = get_code_prediction(codes_true=codes, codes_pred=pomdp_codes_pred, sfreq=params['sfreq'],
                                           code_len=code_len, t_min=t_start, t_max=t_end)

                partial_score = accuracy_score(y_cv, pred)
                print(f"    Partial score for data from {t_start} to {t_end}: {partial_score}")
                y_pred.extend(pred)

            cal_score = accuracy_score(y_long, y_pred)
            print('')
            print(f"Test data was predicted with an acc of {cal_score}")

            conf_matrix = confusion_matrix(y_long, y_pred, normalize='true')
            print('')
            print("Confusion matrix created")
            print('Confusion matrix:')
            print(conf_matrix)

            # If the matrix has zero values and margins are activated, add margins
            if add_margins and np.count_nonzero(conf_matrix == 0):
                score_dict[sub][iter_name]['raw_conf_matrix'] = conf_matrix
                conf_matrix = add_safety_margin(conf_matrix, mixing_coef=mixing_coef)
                print('')
                print('Zero values found in the conf matrix, safety margins added')
                print('New confusion matrix:')
                print(conf_matrix)

            score_dict[sub][iter_name]['cal_score'] = cal_score
            score_dict[sub][iter_name]['conf_matrix'] = conf_matrix

            # POMDP initialization
            all_states = [BCIState(int(state)) for state in label_id]
            init_true_state = random.choice(all_states)  # Random initial state

            # Get initial belief (uniform)
            n_states = len(all_states)
            init_belief = pomdp_py.Histogram({state: 1 / n_states for state in all_states})  # Uniform initial belief

            vep_problem = BCIProblem(init_belief=init_belief, init_true_state=init_true_state,
                                     n_class=n_states, features=conf_matrix, discretization='conf_matrix',
                                     hit_reward=hit_reward, miss_cost=miss_cost, wait_cost=wait_cost)
            print('')
            print("POMDP instance created!")

            sarsop_time = datetime.now()
            policy = sarsop(vep_problem.agent, pomdpsol_path=sarsop_path, discount_factor=gamma,
                            timeout=300, memory=4096, precision=0.001, remove_generated_files=True,
                            pomdp_name=f'temp-pomdp-{sub}-{sarsop_time}')

            # Simulate POMDP using the policy
            print('')
            print('-' * 70)
            print(f"POMDP SIMULATION STARTS")
            print('-' * 70)

            n_trials, _, _ = X_test.shape
            total_reward = 0
            false_positives = 0
            misses = 0
            total_time = 0.
            beliefs = []

            for trial_n in range(n_trials):

                # Separate next trial and label
                next_trial = X_test[trial_n, :, :]
                next_label = int(y_test[trial_n])

                # Convert them to window and normalize
                win_data, win_label = epoch_to_window(next_trial, next_label, sfreq=params['sfreq'], codes=codes,
                                                      win_len=code_win_len, win_format='channels_last',
                                                      return_code_len=False)

                win_data /= cal_std + 1e-8

                # Predict the code (take the first column only)
                code_pred = pomdp_clf.predict(win_data, batch_size=64)[:, 0]

                # Set belief to uniform, in case last trial ended without a decision
                vep_problem.agent.set_belief(init_belief)

                # Set the true state as the env state
                true_state = BCIState(next_label)
                vep_problem.env.apply_transition(true_state)

                print('')
                print(f'TRIAL {trial_n} (true state {true_state})')
                print('-' * 20)

                # For every time step...
                for step_n in range(total_steps):
                    cur_belief = [vep_problem.agent.cur_belief[st] for st in vep_problem.agent.cur_belief]
                    print('')
                    print(f'  STEP {step_n}')
                    print('  Current belief:')
                    print(f'  {cur_belief}')

                    # Get your action and execute it
                    action = policy.plan(vep_problem.agent)
                    reward = vep_problem.env.state_transition(action, execute=False)
                    print('')
                    print(f'  Action: {action.name}')
                    print(f'  Reward: {reward}')

                    # Add your reward
                    total_reward += reward[-1]
                    if reward[-1] == miss_cost:
                        false_positives += 1

                    # Go to next trial if action is taken
                    if action.name != 'a_wait':
                        decision_time = t_end
                        total_time += decision_time
                        beliefs.append((action, cur_belief))
                        print('')
                        print(f'Action {action.name} selected. Trial ended.')
                        print(f'Decision took {decision_time}s')
                        break

                    # Predict and get observation
                    t_start = time_steps[step_n]
                    try:
                        t_end = time_steps[step_n + step_interval]
                    except IndexError:
                        total_time += epoch_len
                        misses += 1
                        print('No decision was taken...')
                        break

                    # The code prediction returns a list of a single label that is a float, so we convert to int and index
                    pred = get_code_prediction(codes_true=codes, codes_pred=code_pred, sfreq=params['sfreq'],
                                               code_len=code_len, t_min=t_start, t_max=t_end)

                    observation = BCIObservation(int(pred[0]))
                    print(f'  Observation: {observation.name}')

                    # Belief update
                    new_belief = pomdp_py.update_histogram_belief(vep_problem.agent.cur_belief,
                                                                  action, observation,
                                                                  vep_problem.agent.observation_model,
                                                                  vep_problem.agent.transition_model,
                                                                  static_transition=False)
                    vep_problem.agent.set_belief(new_belief)

            # Save results for this POMDP
            score_dict[sub][iter_name]['total_reward'] = total_reward
            score_dict[sub][iter_name]['FP'] = false_positives
            score_dict[sub][iter_name]['misses'] = misses
            score_dict[sub][iter_name]['avgtime'] = total_time / n_trials
            score_dict[sub][iter_name]['acc'] = (n_trials - (false_positives + misses)) / n_trials

            # Store metadata for this POMDP
            metadata_dict[iter_name] = {'template_len': epoch_len,
                                        'sfreq': params['sfreq'],
                                        'window_len': slice_len,
                                        'time_step': time_step,
                                        'gamma': gamma,
                                        'hit_reward': hit_reward,
                                        'miss_cost': miss_cost,
                                        'wait_cost': wait_cost,
                                        'obs_matrix_regu': add_margins,
                                        'test_n': test_n,
                                        'cv_n': cv_n,
                                        'algorithm': algo}

        # Get score of regular algo (without POMDP on X_train)
        data_train, labels_train, code_len = epoch_to_window(X_train, y_train, sfreq=params['sfreq'], codes=codes,
                                                             win_len=code_win_len, win_format='channels_last',
                                                             return_code_len=True)

        data_test, labels_test = epoch_to_window(X_test, y_test, sfreq=params['sfreq'], codes=codes,
                                                 win_len=code_win_len, win_format='channels_last',
                                                 return_code_len=False)

        X_std_train = data_train.std(axis=0)
        data_train /= X_std_train + 1e-8
        data_test /= X_std_train + 1e-8

        baseline_clf = fit_clf(data_train, labels_train, algorithm=algo)

        # Predict the codes of validation data (split in 10 to avoid OOM from the GPU)
        baseline_codes_pred = []
        bits_per_class = int(len(data_test) / n_class)
        pred_splits = 11
        for i in range(pred_splits):
            if i == pred_splits - 1:
                data_slice = data_test[bits_per_class * i:]
            else:
                data_slice = data_test[bits_per_class * i: bits_per_class * (i + 1)]

            pred = baseline_clf.predict(data_slice, batch_size=64)
            baseline_codes_pred.extend(pred[:, 0])

        baseline_codes_pred = np.array(baseline_codes_pred)  # In window format

        for clf_method in clf_methods:
            # Predict data on current time step and add to pred array. This goes back to non-window format
            score_dict[sub][f'baseline_{clf_method}'] = {}

            if clf_method == 'cvep_pvalue':
                labels_pred, mean_time = get_code_prediction(codes_true=codes, codes_pred=baseline_codes_pred,
                                                             sfreq=params['sfreq'], code_len=code_len,
                                                             t_min=0., t_max=0.5, incremental=True)
            elif clf_method == 'cvep_cumul':
                labels_pred, mean_time = make_preds_accumul_aggresive(codes_true=codes, codes_pred=baseline_codes_pred,
                                                                      sfreq=params['sfreq'], code_len=code_len)
            elif clf_method == 'cvep_static':  # Predict on the first 0.5s
                labels_pred = get_code_prediction(codes_true=codes, codes_pred=baseline_codes_pred, sfreq=params['sfreq'],
                                                  code_len=code_len, t_min=0, t_max=0.5)
                mean_time = 0.5
            else:
                raise NameError('Invalid classification method!')

            # Get acc, FPs, misses, etc.
            cvep_acc = accuracy_score(y_test, labels_pred)
            cvep_misses = np.count_nonzero(labels_pred == -1)
            cvep_misses_score = cvep_misses / len(labels_pred)
            cvep_fp_score = 1 - cvep_acc - cvep_misses_score
            cvep_fp = int(cvep_fp_score * len(labels_pred))
            cvep_avg_time = mean_time

            score_dict[sub][f'baseline_{clf_method}']['acc'] = cvep_acc
            score_dict[sub][f'baseline_{clf_method}']['misses'] = cvep_misses
            score_dict[sub][f'baseline_{clf_method}']['FP'] = cvep_fp
            score_dict[sub][f'baseline_{clf_method}']['avgtime'] = cvep_avg_time

# Flip score_dict: it is saved as subs -> iterations -> metrics, and we want iterations -> subs -> metrics
flip_score_dict = defaultdict(dict)
for sub, iterations in score_dict.items():
    for iter_name, metrics in iterations.items():
        flip_score_dict[iter_name][sub] = metrics

for iter_name, scores in flip_score_dict.items():
    if 'baseline' in iter_name:
        metadata = {'template_len': epoch_len + code_win_len,
                    'sfreq': params['sfreq'],
                    'window_len': slice_len,
                    'time_step': '--',
                    'gamma': '--',
                    'hit_reward': '--',
                    'miss_cost': '--',
                    'wait_cost': '--',
                    'obs_matrix_regu': '--',
                    'regu_type': '--',
                    'test_n': test_n,
                    'cv_n': '--',
                    'algorithm': iter_name}
    else:
        metadata = metadata_dict[iter_name]

    results_filename = os.path.join(results_path, f'{iter_name}.hdf5')
    results_df = pd.DataFrame.from_dict(scores).T
    save_results(results_filename, results_df, metadata)

# Grand results
metrics = ['FP', 'misses', 'avgtime', 'acc']  # Common metrics between POMDP and CVEP
clf_methods.append('POMDP')

for iter_name, scores in flip_score_dict.items():
    # Create shared metrics
    grand_results[iter_name] = {f'grand_{metric}': 0 for metric in metrics}
    # Add number of trials
    grand_results[iter_name]['n_trials'] = n_trials
    # Add reward
    grand_results[iter_name]['grand_reward'] = 0

    print('')
    print(iter_name)
    print('-' * 30)
    print('')

    # Print and add each sub's metrics to the totals
    for sub in sub_list:
        print(f'  SUB {sub}:')

        if 'baseline' not in iter_name:
            sub_reward = flip_score_dict[iter_name][sub]['total_reward']
            print(f'    Total reward (POMDP): {sub_reward}')
            grand_results[iter_name]['grand_reward'] += sub_reward

        for metric in metrics:
            metric_method = flip_score_dict[iter_name][sub][f'{metric}']
            print(f'    {metric}: {metric_method}')
            grand_results[iter_name][f'grand_{metric}'] += metric_method

        print('')

    # Print and save the results for this iteration (dataset, model, etc.)
    print(f'FINAL METRICS ({iter_name})')
    print('-' * 70)
    for metric_name, final_metric in grand_results[iter_name].copy().items():
        if 'acc' in metric_name:
            print('')
        print(f'{metric_name} (all_subjects): {final_metric}')

        if 'avgtime' in metric_name:
            continue
        else:
            avg_metric = final_metric / len(sub_list)
            print(f'Average {metric_name} per subbject (out of {n_trials} trials): {avg_metric}')
            grand_results[iter_name][f'avg_{metric_name}'] = avg_metric

# At the end of time, or the beginning, who knows at this point... save the results for comparison
grand_df = pd.DataFrame(grand_results)
grand_df.to_csv(grand_results_filename)

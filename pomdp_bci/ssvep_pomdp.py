"""
SSVEP-POMDP implementation

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import itertools
import json
import os
import pomdp_py
import random

import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pomdp_py import sarsop

from pomdp_bci.problem import BCIProblem
from pomdp_bci.domain import BCIState, BCIObservation
from pomdp_bci.utils import TRCA
from pomdp_bci.utils import load_data, add_safety_margin, save_results

# Path variables - put your paths here or export them to your environment 
data_path = os.environ.get('SSVEP')
sarsop_path = os.environ.get('SARSOP')

time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
results_path = os.path.join(os.getcwd(), f'results/SSVEP_{time}')
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Data variables
init_delay = 0.13  # In seconds, 0.13 in the original TRCA paper
epoch_len = 1.  # In seconds
n_fbands = 5  # Parameter for filterbank

test_n = 2  # Number of trials (per class) used for POMDP simulation
cv_n = 5  # Number of trials (per class) used for Confusion matrix calculation
add_margins = True  # Minimize the impact of zeroes and big numbers on the conf matrix
mixing_coef = 0.3  # Parameter for the margins operation

# POMDP variables
hit_reward = 10
miss_costs = [-100, -1000]
wait_cost = -1
gamma = 0.99

# Analysis variables
slice_steps = [(0.5, 0.1)]  # (slice_len, time_step)
datasets = ['ssvep']

# Big results filename
grand_results_filename = os.path.join(results_path, f'comparison_ssvep_{time}.csv')
grand_results = {}

for dataset, (slice_len, time_step) in itertools.product(datasets, slice_steps):
    # Load dataset-specific parameters
    with open(f'{config/dataset}.json', 'r') as dataset_params:
        params = json.loads(dataset_params.read())

    downsample = int(params['sfreq'] / 250)  # After downsampling, sfreq should be 250Hz

    sub_list = [1]  #[sub_n + 1 for sub_n in range(params['n_subs'])]
    score_dict = {sub: {} for sub in sub_list}
    metadata_dict = {}

    # results_filename = os.path.join(results_path, f'{dataset}_{study_type}_{margin_method}_{time}.hdf5')

    for sub in sub_list:
        # Load the data (preprocessed and epoched)
        epochs = load_data(subject=sub, dataset=dataset, eeg_path=data_path, ch_keep=params["ch_slice"])

        labels = epochs.events[..., -1]
        labels -= np.min(labels)

        t_min = int(init_delay * params['sfreq']) + 1  # In samples
        t_max = int(t_min + epoch_len * params['sfreq'])  # Also in samples

        data = epochs.get_data()
        data = data[..., t_min:t_max]
        print(f"Selecting data from epoch {t_min} to {t_max}, "
              f"corresponding to {init_delay} and {init_delay + epoch_len}s")

        # Remove first event if there is 'boundary' on event_id
        if 'boundary' in epochs.event_id.keys():
            labels = labels[1:, ...]
            data = data[1:, ...]

        # Analysis
        freq_list = [float(freq) for freq in epochs.event_id.keys() if freq != 'boundary']

        label_id, label_count = np.unique(labels, return_counts=True)
        n_class = len(label_id)
        print('')
        print(f"Using freqs: {freq_list}")
        print(f"Using labels {label_id} with number of trials {label_count}")

        # Tran-test cross-validation
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=(n_class * test_n),
                                                            random_state=42, stratify=labels)

        # Confusion matrix cross-validation
        cv_splitter = StratifiedShuffleSplit(n_splits=12, test_size=(n_class * cv_n), random_state=42)

        # Try every combination of calibration and test to get the best confusion matrix
        best_score = 0
        best_split = 0
        conf_matrix = 0

        for split_n, (train_index, test_index) in enumerate(cv_splitter.split(X_train, y_train)):
            X_cal, X_cv = X_train[train_index], X_train[test_index]
            y_cal, y_cv = y_train[train_index], y_train[test_index]

            # Metrics
            print('')
            print(f"Split {split_n}")
            print(f"Creating confusion matrix using an epoch length of {slice_len}s")
            time_steps = [i * time_step for i in range(20) if i * time_step <= epoch_len]
            step_interval = int(slice_len / time_step)
            n_steps = len(time_steps)
            y_pred = []
            y_long = []
            models = {}  # Fit one TRCA model for every time step

            for step_n in range(n_steps):
                t_start = time_steps[step_n]
                try:
                    t_end = time_steps[step_n + step_interval]
                    y_long.extend(list(y_cv))  # Labels are the same no matter the slice
                except IndexError:
                    continue

                # Create TRCA and fit with sliced data
                trca = TRCA(sfreq=params['sfreq'], n_fbands=n_fbands, peaks=freq_list,
                            downsample=downsample, method='original',
                            is_ensemble=True)

                sample_start = int(t_start * params['sfreq'])
                sample_end = int(t_end * params['sfreq'])
                trca.fit(X_cal[..., sample_start:sample_end], y_cal)
                print('')
                print(f"    TRCA was fit from {t_start} to {t_end} ({sample_start} to {sample_end})")

                # Predict data on current time step and add to pred array
                pred = trca.predict(X_cv[..., sample_start:sample_end])
                partial_score = accuracy_score(y_cv, pred)
                print(f"    Partial score for data from {t_start} to {t_end}: {partial_score}")
                y_pred.extend(pred)

                # Save the model for later
                models[step_n] = trca

            score = accuracy_score(y_long, y_pred)
            print('')
            print(f"Test data was predicted with an acc of {score}")

            cf = confusion_matrix(y_long, y_pred, normalize='true')
            print("Confusion matrix created")

            if score >= best_score:
                best_score = score
                best_split = split_n + 1
                best_model = models
                conf_matrix = cf

        # POMDP starts here
        for miss_cost in miss_costs:
            # Create dictionary entry for this iteration
            iter_name = f'{dataset}-slice{slice_len}_step{time_step}_cost{miss_cost}_gamma{gamma}'
            score_dict[sub][iter_name] = {}

            # Prinf conf matrix info
            print('')
            print(f'CV ended, best score: {best_score} (split {best_split})')
            print('Best confusion matrix:')
            print(conf_matrix)

            score_dict[sub][iter_name]['cal_score'] = best_score

            # If the matrix has zero values and margins are activated, add margins
            # (except if the conf matrix is the identity matrix)
            if (conf_matrix == np.identity(n_class)).all():
                pass
            elif add_margins and np.count_nonzero(conf_matrix == 0):
                score_dict[sub][iter_name]['raw_conf_matrix'] = conf_matrix
                conf_matrix = add_safety_margin(conf_matrix, mixing_coef=mixing_coef)
                print('')
                print('Zero values found in the conf matrix, safety margins added')
                print('New confusion matrix:')
                print(conf_matrix)

            score_dict[sub][iter_name]['conf_matrix'] = conf_matrix

            # Print some data about the model
            print('')
            print('Creating confusion matrix, POMDP parameters:')
            print(f'  Window length: {slice_len}')
            print(f'  Time step: {time_step}')
            print(f'  Cost for misses: {miss_cost}')
            print(f'  Discount factor: {gamma}')

            # POMDP initialization
            all_states = [BCIState(int(state)) for state in label_id]
            init_true_state = random.choice(all_states)  # Random initial state
            n_states = len(all_states)
            init_belief = pomdp_py.Histogram({state: 1 / n_states for state in all_states})  # Uniform initial belief

            vep_problem = BCIProblem(init_belief=init_belief, init_true_state=init_true_state,
                                     n_class=n_states, features=conf_matrix, discretization='conf_matrix',
                                     miss_cost=miss_cost)
            print('')
            print("POMDP instance created!")

            policy = sarsop(vep_problem.agent, pomdpsol_path=sarsop_path, discount_factor=gamma,
                            timeout=60, memory=2048, precision=0.001, remove_generated_files=True)

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

                # Set belief to uniform, in case last trial ended without a decision
                vep_problem.agent.set_belief(init_belief)

                # Set the true state as the env state
                true_state = BCIState(next_label)
                vep_problem.env.apply_transition(true_state)

                print('')
                print(f'TRIAL {trial_n} (true state {true_state})')
                print('-' * 20)

                # For every time step...
                for step_n in range(n_steps):
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

                    # Get prediction from the corresponding model
                    sample_start = int(t_start * params['sfreq'])
                    sample_end = int(t_end * params['sfreq'])

                    pred = int(best_model[step_n].predict(next_trial[..., sample_start:sample_end])[0])
                    observation = BCIObservation(pred)
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
            score_dict[sub][iter_name]['POMDP_acc'] = (n_trials - (false_positives + misses)) / n_trials

            # Baseline
            for window_len in [0.5, 1]:
                trca = TRCA(sfreq=params['sfreq'], n_fbands=n_fbands, peaks=freq_list,
                            downsample=downsample, method='original',
                            is_ensemble=True)

                sample_end = int(window_len * params['sfreq'])
                trca.fit(X_train[..., :sample_end], y_train)

                pred = trca.predict(X_test[..., :sample_end])
                score = accuracy_score(y_test, pred)
                score_dict[sub][iter_name][f'trca_{window_len}'] = score

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
                                        'regu_type': margin_method,
                                        'test_n': test_n,
                                        'cv_n': cv_n}

# Flip score_dict: it is saved as subs -> iterations -> metrics, and we want iterations -> subs -> metrics
flip_score_dict = defaultdict(dict)
for sub, iterations in score_dict.items():
    for iter_name, metrics in iterations.items():
        flip_score_dict[iter_name][sub] = metrics

# Save results of individual iterations in a hdf5 file
for iter_name, scores in flip_score_dict.items():
    metadata = metadata_dict[iter_name]

    results_filename = os.path.join(results_path, f'{iter_name}.hdf5')
    results_df = pd.DataFrame.from_dict(scores).T
    save_results(results_filename, results_df, metadata)

# Grand results
metrics = ['FP', 'misses', 'avgtime', 'POMDP_acc', 'trca_0.5', 'trca_1']

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


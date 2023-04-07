"""
MI-POMDP implementation using the public dataset BNCI2014001 from MOABB
(http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html)

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
from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery
from sklearn.preprocessing import LabelEncoder

from pomdp_py import sarsop
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from pomdp_bci.problem import BCIProblem
from pomdp_bci.domain import BCIState, BCIObservation
from pomdp_bci.utils import save_results, add_safety_margin

# Path variables
sarsop_path = os.environ.get('SARSOP') 

time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
results_path = os.path.join(os.getcwd(), f'results/MI_{time}')
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Data variables
init_delay = 0.  # In seconds
epoch_len = 4  # In seconds
cv_n = 4  # Number runs to use for confusion matrix creation
test_n = 2  # Number of runs used for POMDP simulation
add_margins = True  # Minimize the impact of zeroes and big numbers on the conf matrix
mixing_coef = 0.3  # Parameter for margin operation
margins_added = False  # Flag for margins

# POMDP variables
hit_reward = 10
miss_costs = [-100, -1000]
wait_cost = -1

# Since we know the number of steps for each POMDP trial, we can set the gamma automatically instead of heuristically,
# as the value we want for it to be when elevated to the power of the last time step. Knowing that the value formula
# for the POMDP weights future values as gamme ** t, we can specify the value we want gamma to have at t_max
gamma_horizon = 0.25

# Analysis variables
slice_steps = [(3.0, 0.1)]  # (slice_len, time_step)
datasets = ['MI_cv-cross']
algos = ['RG+LR']  # Riemannian Geometry + Logistic Regression
baseline_epoch_lens = [3, 4]  # In seconds

# Big results filename (saves on the results folder directly)
grand_results_filename = os.path.join(results_path, f'../comparison_mi_{time}.csv')
grand_results = {}  # Keys are iterations, hold results for all subjects
dataset_results = {}

for dataset_name, algo in itertools.product(datasets, algos):
    # Load dataset-specific parameters
    modality, analysis = dataset_name.split('_')
    with open(f'{modality}.json', 'r') as dataset_params:
        params = json.loads(dataset_params.read())

    if 'MI' in dataset_name:
        dataset = BNCI2014001()
        paradigm = LeftRightImagery()

    sub_list = [sub_n + 1 for sub_n in range(params['n_subs'])]
    score_dict = {sub: {} for sub in sub_list}
    metadata_dict = {}

    for sub in sub_list:
        print('')
        print(f'Sub {sub}:')
        print('*' * 10)
        # Load the data (preprocessed and epoched)
        data, labels, metadata = paradigm.get_data(dataset, [sub], return_epochs=False)

        # Labels need encoding
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        label_id, label_count = np.unique(labels, return_counts=True)

        # Slice returned epochs from init_delay to epoch_len
        t_min = int(init_delay * params['sfreq']) + 1  # In samples
        t_max = int(t_min + epoch_len * params['sfreq'])  # Also in samples

        data = data[..., t_min:t_max]
        print(f"Selecting data from epoch {t_min} to {t_max}, "
              f"corresponding to {init_delay} and {init_delay + epoch_len}s")

        # Extract runs and sessions for data splitting
        runs = np.array([int(run.split('_')[-1]) for run in metadata.run.values])
        sessions = metadata.session.values

        # Divide the data in calibration (used to train the algorithm), cv (used to produce the confusion matrix)
        # and test (used to play the POMDP model at the end)
        if "cross" in analysis:
            # Separate the entire first session as calibration data
            X_T, y_T = data[sessions == 'session_T'], labels[sessions == 'session_T']
            X_E, y_E = data[sessions == 'session_E'], labels[sessions == 'session_E']

            # Then divide the second session between cv and test
            runs_session = runs[sessions == 'session_E']
            if 'cv' in analysis:
                # Whole first session for calibration
                X_cal, y_cal = data[sessions == 'session_T'], labels[sessions == 'session_T']

                # Second session split between cv and test
                X_cv, y_cv = X_E[runs_session < cv_n], y_E[runs_session < cv_n]
                X_test, y_test = X_E[runs_session >= cv_n], y_E[runs_session >= cv_n]

            elif 'hybrid' in analysis:
                run_idx = np.unique(runs_session)
                last_run = np.argmax(run_idx)
                cv_block = cv_n / 2

                # For calibration use the first runs of the first block except the cv_block last
                X_cal, y_cal = X_T[runs_session <= last_run - cv_block], \
                               y_T[runs_session <= last_run - cv_block]

                # Get last cv_n / 2 runs from the first session and the same number at the beginning of the second
                # for CV
                X_cv_t, y_cv_t = X_T[runs_session > last_run - cv_block], \
                                 y_T[runs_session > last_run - cv_block]

                X_cv_e, y_cv_e = X_E[runs_session < cv_block], y_E[runs_session < cv_block]

                X_cv, y_cv = np.concatenate([X_cv_t, X_cv_e], axis=0), np.concatenate([y_cv_t, y_cv_e], axis=0)

                # Lastly, get all the second session except the first cv_block for testing
                X_test, y_test = X_E[runs_session >= cv_block], y_E[runs_session >= cv_block]

            print('')
            print('Data was divided (cross-session)')
            print(f'  Calibration data: {len(y_cal)} trials (first session)')
            print(f'  CV data: {len(y_cv)} trials (second session)')
            print(f'  Test data: {len(y_test)} trials (second session)')
            print('')
            print(f'Number of classes: {len(np.unique(labels))}')

        trials_per_run = X_cal.shape[0] / len(np.unique(runs))
        print(f'Trials per run: {trials_per_run}')

        # Analysis
        if algo == "RG+LR":
            cov = Covariances()
            tan = TangentSpace()
            clf = LogisticRegression(solver='lbfgs')

            # Estimate covariance matrices and tangent space
            feat_cal = cov.transform(X_cal)
            tan.fit(feat_cal)

            feat_cal = tan.transform(feat_cal)

            clf.fit(feat_cal, y_cal)

        # Once the model is ready, create a POMDP using different parameters for length of the window to be used,
        # the time step, and the cost for misses
        for (slice_len, time_step), miss_cost, margin_method in itertools.product(slice_steps, miss_costs):
            # Get the number of steps the model has to simulate the policy according to epoch_len, slice_len
            # and time_step
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
            iter_name = f'{dataset_name}_{algo}-slice{slice_len}_step{time_step}_cost{miss_cost}_gamma{gamma}_regu{margin_method}'
            score_dict[sub][iter_name] = {}

            # Get total steps and step interval
            time_steps = list(np.arange(0, epoch_len + 0.01, time_step))
            total_steps = len(time_steps)
            step_interval = int(slice_len / time_step)
            y_pred = []  # All predictions of all slices
            y_long = []  # Regular labels repeated once per time step

            # Get partial score for every time step
            print('')
            for step_n in range(total_steps):
                # Get start and end times
                t_start = time_steps[step_n]
                try:
                    t_end = time_steps[step_n + step_interval]
                    y_long.extend(list(y_cv))  # Labels are the same no matter the slice
                except IndexError:
                    continue

                # Transform sliced data to tangent space
                sample_start = int(t_start * params['sfreq'])
                sample_end = int(t_end * params['sfreq'])
                feat_slice = cov.transform(X_cv[..., sample_start:sample_end])
                feat_slice = tan.transform(feat_slice)

                slice_pred = clf.predict(feat_slice)
                partial_score = accuracy_score(y_cv, slice_pred)
                print(f"    Partial score for data from {t_start} to {t_end}: {partial_score}")
                y_pred.extend(slice_pred)

            # General cv score
            cal_score = accuracy_score(y_long, y_pred)
            print('')
            print(f"Test data was predicted with an acc of {cal_score}")

            conf_matrix = confusion_matrix(y_long, y_pred, normalize='true')
            print('')
            print("Confusion matrix created")
            print(f'  Confusion Matrix ({iter_name}):')
            print("\t" + str(conf_matrix).replace('\n', '\n\t'))
            print()

            # If the matrix has zero values and margins are activated, add margins
            if add_margins:  # and np.count_nonzero(conf_matrix == 0):
                score_dict[sub][iter_name]['raw_conf_matrix'] = conf_matrix
                score_dict[sub][iter_name]['mixing_coef'] = mixing_coef
                margins_added = True

                conf_matrix = add_safety_margin(conf_matrix, mixing_coef=mixing_coef)
                print('')
                print('Zero values found in the conf matrix, safety margins added')
                print('New confusion matrix:')
                print("\t" + str(conf_matrix).replace('\n', '\n\t'))
            
            score_dict[sub][iter_name]['cal_score'] = cal_score
            score_dict[sub][iter_name]['conf_matrix'] = conf_matrix

            # POMDP initialization
            all_states = [BCIState(int(state)) for state in label_id]
            init_true_state = random.choice(all_states)  # Random initial state

            # Get initial belief (uniform)
            n_states = len(all_states)
            init_belief = pomdp_py.Histogram({state: 1 / n_states for state in all_states})  # Uniform initial belief

            mi_problem = BCIProblem(init_belief=init_belief, init_true_state=init_true_state,
                                    n_class=n_states, features=conf_matrix, discretization='conf_matrix',
                                    hit_reward=hit_reward, miss_cost=miss_cost, wait_cost=wait_cost)

            print('')
            print("POMDP instance created!")

            sarsop_time = datetime.now()
            policy = sarsop(mi_problem.agent, pomdpsol_path=sarsop_path, discount_factor=gamma,
                            timeout=120, memory=4096, precision=0.001, remove_generated_files=True,
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
                next_trial = X_test[trial_n, ...]
                next_trial = next_trial[np.newaxis, ...]

                next_label = int(y_test[trial_n])

                # Set belief to uniform, in case last trial ended without a decision
                mi_problem.agent.set_belief(init_belief)

                # Set the true state as the env state
                true_state = BCIState(next_label)
                mi_problem.env.apply_transition(true_state)

                trial_beliefs = []

                print('')
                print(f'TRIAL {trial_n} (true state {true_state})')
                print('-' * 20)

                # Use a list of all time steps from 0 and take the slice by skipping n steps in
                # the list according to step_interval. Stop when your time_end index is out of the list
                for step_n in range(total_steps):
                    cur_belief = [mi_problem.agent.cur_belief[st] for st in mi_problem.agent.cur_belief]
                    print('')
                    print(f'  STEP {step_n}')
                    print('  Current belief:')
                    print(f'  {cur_belief}')

                    # Get your action and execute it
                    action = policy.plan(mi_problem.agent)
                    reward = mi_problem.env.state_transition(action, execute=False)
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
                        trial_beliefs.append((cur_belief, action.name, '--'))
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

                    # Slice and transform data (using tangent space estimation from calibration
                    sample_start = int(t_start * params['sfreq'])
                    sample_end = int(t_end * params['sfreq'])
                    next_feat = cov.transform(next_trial[..., sample_start:sample_end])
                    next_feat = tan.transform(next_feat)

                    pred = int(clf.predict(next_feat)[0])
                    observation = BCIObservation(pred)
                    trial_beliefs.append((cur_belief, action.name, observation.name))
                    print(f'  Observation: {observation.name}')

                    # Belief update
                    new_belief = pomdp_py.update_histogram_belief(mi_problem.agent.cur_belief,
                                                                  action, observation,
                                                                  mi_problem.agent.observation_model,
                                                                  mi_problem.agent.transition_model,
                                                                  static_transition=False)
                    mi_problem.agent.set_belief(new_belief)

                # Add trial beliefs to list of all beliefs
                beliefs.append(trial_beliefs)

            # Save results for this POMDP
            score_dict[sub][iter_name]['total_reward'] = total_reward
            score_dict[sub][iter_name]['FP'] = false_positives
            score_dict[sub][iter_name]['misses'] = misses
            score_dict[sub][iter_name]['avgtime'] = total_time / n_trials
            score_dict[sub][iter_name]['acc'] = (n_trials - (false_positives + misses)) / n_trials
            score_dict[sub][iter_name]['beliefs'] = beliefs

            # Store metadata for this POMDP
            metadata_dict[iter_name] = {'template_len': epoch_len,
                                        'sfreq': params['sfreq'],
                                        'window_len': slice_len,
                                        'time_step': time_step,
                                        'gamma': gamma,
                                        'hit_reward': hit_reward,
                                        'miss_cost': miss_cost,
                                        'wait_cost': wait_cost,
                                        'margins_added': margins_added,
                                        'mixing_coef': mixing_coef,
                                        'regu_type': margin_method,
                                        'test_n': len(y_test),
                                        'cv_n': len(y_cv),
                                        'algorithm': algo,
                                        'analysis': dataset_name}

            # Get score on regular algo (without POMDP on X_train). We use the same data for training and
            # the same data for testing (omitting data for cv as it would make comparisons more difficult)
            for baseline_epoch_len in baseline_epoch_lens:
                # Slice the data for the len we want
                sample_start = 0
                sample_end = baseline_epoch_len * params['sfreq']
                baseline_cal = X_cal[..., sample_start:sample_end]
                baseline_test = X_test[..., sample_start:sample_end]

                # Get features and fit
                feat_cal = cov.transform(baseline_cal)
                feat_cal = tan.transform(feat_cal)

                clf.fit(feat_cal, y_cal)

                # Predict

                score_dict[sub][f'baseline_{baseline_epoch_len}_{dataset_name}'] = {}

                feat_test = cov.transform(baseline_test)
                feat_test = tan.transform(feat_test)
                labels_pred = clf.predict(feat_test)
                mean_time = baseline_epoch_len

                # Get acc, FPs, misses, etc.
                mi_acc = accuracy_score(y_test, labels_pred)
                mi_misses = np.count_nonzero(labels_pred == -1)
                mi_misses_score = mi_misses / len(labels_pred)
                mi_fp_score = 1 - mi_acc - mi_misses_score
                mi_fp = int(mi_fp_score * len(labels_pred))
                mi_avg_time = mean_time

                score_dict[sub][f'baseline_{baseline_epoch_len}_{dataset_name}']['acc'] = mi_acc
                score_dict[sub][f'baseline_{baseline_epoch_len}_{dataset_name}']['misses'] = mi_misses
                score_dict[sub][f'baseline_{baseline_epoch_len}_{dataset_name}']['FP'] = mi_fp
                score_dict[sub][f'baseline_{baseline_epoch_len}_{dataset_name}']['avgtime'] = mi_avg_time

    # Flip score_dict: it is saved as subs -> iterations -> metrics, and we want iterations -> subs -> metrics
    flip_score_dict = defaultdict(dict)
    for sub, iterations in score_dict.items():
        for iter_name, metrics in iterations.items():
            flip_score_dict[iter_name][sub] = metrics

    for iter_name, scores in flip_score_dict.items():
        if 'baseline' in iter_name:
            metadata = {'template_len': epoch_len,
                        'sfreq': params['sfreq'],
                        'window_len': '--',
                        'time_step': '--',
                        'gamma': '--',
                        'hit_reward': '--',
                        'miss_cost': '--',
                        'wait_cost': '--',
                        'margins_added': '--',
                        'margin_max': '--',
                        'regu_type': '--',
                        'test_n': len(y_test),
                        'cv_n': '--',
                        'algorithm': iter_name,
                        'analysis': dataset_name}
        else:
            metadata = metadata_dict[iter_name]

        results_filename = os.path.join(results_path, f'{iter_name}.hdf5')
        results_df = pd.DataFrame.from_dict(scores).T
        save_results(results_filename, results_df, metadata)

        dataset_results[dataset_name] = flip_score_dict

# Grand results
metrics = ['FP', 'misses', 'avgtime', 'acc']  # Common metrics between POMDP and CVEP

for dataset in datasets:
    iter_dict = dataset_results[dataset]

    for iter_name, scores in iter_dict.items():
        # Create shared metrics
        grand_results[iter_name] = {f'grand_{metric}': 0 for metric in metrics}
        # Add number of subjects
        grand_results[iter_name]['n_sub'] = len(sub_list)
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
                sub_reward = iter_dict[iter_name][sub]['total_reward']
                print(f'    Total reward (POMDP): {sub_reward}')
                grand_results[iter_name]['grand_reward'] += sub_reward

            for metric in metrics:
                metric_method = iter_dict[iter_name][sub][f'{metric}']
                print(f'    {metric}: {metric_method}')
                grand_results[iter_name][f'grand_{metric}'] += metric_method

            print('')

        # Print and save the results for this iteration (dataset, model, etc.) inside the folder corresponding
        # to this analysis
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

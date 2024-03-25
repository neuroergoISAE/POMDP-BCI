"""
Experiment classes for POMDP-BCI simulation

Author: Juan Jesus Torre Tresols
mail: juan-jesus.torre-tresols@isae-supaero.fr
"""

import itertools
import json
import math
import os
import random
import pickle

import mne
import pomdp_py

import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import datetime
from pomdp_py import sarsop
from pomdp_py.utils.interfaces.conversion import AlphaVectorPolicy
from sklearn.metrics import accuracy_score, confusion_matrix

from bci_pomdp.problem import BaseProblem, TDProblem
from bci_pomdp.domain import BCIState, TDState, BCIObservation


class Experiment:
    """
    Class that contains common methods for all POMDP-BCI experiments, including data loading,
    decoding algorithm calibration, POMDP initialization, POMDP simulation and saving results.

    Parameters
    ----------

    epoch_len: int or float
        Duration of the epochs to analyze in seconds. This is the maximum duration used for each
        POMDP trial.

    o_cal_n: int
        Number of trials per class to use in order to create the observation model for the POMDP.
        This data is used to get predictions from the fitted decoding algoritm and estimate a confusion
        matrix, which is used as the observation model.

    test_n: int
        Number of trials per class used to simulate the POMDP.

    data_win_lens: float or list of float
        Length (or lengths) of the data slice used for each step of the POMDP model. If a single element
        is passed, it will be converted to a single-element list. If the POMDP model is time-dependent,
        each element represents the minimum duration of the data window, as its length varies.

    time_steps: float or list of float
        Value for the time-step of the POMDP. At each time step of the simulation, the data used will be
        advanced according to this value with respect to the data used on the previous time step. It will
        be converted to list if a single element is passed. If the POMDP model is time-dependent, the length
        of the data window increases by this parameter every step.

    datasets: str or list of str
        Names of the datasets to try for the experiment. It will be converted to list if a single element
        is passed

    algos: str or list of str
        Names of the different decoding algorithms to try for the experiment. It will be converted to list
        if a single element is passed

    baseline_methods: str, list of str or None, default=None
        Names of the baseline methods to compare the POMDP against. It will be converted to list of a single
        element is passed. If None, no baseline will be run.

    init_delay: float
        Initial delay to add to the epochs when slicing the data window that will be used for analyses. This is used
        in VEP-based BCIs, notoriously on TRCA [1], explained as the delay of the human visual pathway to reflect the
        oscilatory activity of the stimuli. According to [1], it should be 0.14 seconds.

    time_dependent: bool or list of bool, default=False
        Determines whether the POMDP associated with the experiment is time-dependent or not. This affects
        the creation of the POMDP problem and how other data_win_lens and time_steps are interpreted. This
        also affects how the model time steps are created for the training of the decoder and POMDP simulation.
        If a single element is passed, it will be converted to a single-element list.

    pomdp_go: bool or list of bool, default=False
        Determines whether the POMDP uses the POMDP-GO exension. This involves that every trial where the agent
        makes a mistake receives an equal penalty, irrespective of the length of the trial. This translates
        in subtracting the accumulated cost from waiting to the cost of making a mistake in every trial.
        NOTE: This is only possible in time-dependent problems.

    downsample: bool, default=True
        If True, data will be downsampled.

    downsample_target: int, default=250
        If downsample is set to True, data will be downsampled to downsample_target. Not used otherwise

    add_margins: bool, default=True
        Whether to smooth the confusion matrix or not. If true, a uniform distribution will be mixed with each
        row of the confusion matrix.

    mix_values: int or list of int, default=0.3
        Mixing parameter used for confusion matrix normalization. If a single value is passed, it will be converted
        to list. See self._add_safety_margin() for more information.

    sarsop_timeout: int, default=180
        Timeout in seconds for the SARSOP solver. Only used if the solver is set to 'sarsop'.

    hit_reward: int, default=10
        POMDP reward for correct decision.

    miss_costs: int or list of int, default=[-100, -1000]
        POMDP reward for incorrect decision. It will be converted to list if a single value is passed.

    wait_cost: int, default=-1
        POMDP reward for the 'wait' action.

    gamma_horizon: float, default=0.25
        The target gamma value at the last time step of the model. Since each trial has a maximum number of
        time steps, the discount factor of the POMDP is decided such as it is equal to the gamma_horizon at
        the last time_step.

    solvers: str or list of str, default='auto'
        Solver or solvers to use for the different POMDP models. Can be 'auto', 'sarsop' or 'pomcp'.
        Using 'auto' will assign a solver to the problem depending on some of its parameters. See
        self._get_solver() for more information. If a single value is passed, it will be converted to list.

    simu_modes: str or list of str, ['data', 'generative'], default=data
        Model to use for POMDP simulation. Data mode takes real EEG data from the testing data and generates
        observations based on the underlying classification algorithm. Generative mode samples from the
        observation model and does not require data.

    config_path: str or path object, default=''
        Path where the .json configuration files are stored. This class assumes that each dataset has its
        own configuration file, containing dataset-relevant parameters such as number of subjects and data
        sampling frequency. The name of the each file must be [dataset].json for each dataset in datasets.

    data_path: str or path object, default=''
        Path where the EEG files are stored. This class assumes all files for all participants of a given
        experiment are on the same folder and follow a common naming convention.

    solver_path: str or path object, default=''
        Path to the POMDP solver. Only needed for SARSOP.

    results_path: str or path object, default=''
        Path where to store the results files. If the path does not exist, it will be created.

    Attributes
    ----------

    fitted: dict of dicts of dicts of bool
        Contains flags for each combination of dataset and algorithm, indicating whether the decoding
        algorithm has been fitted for each of the dataset's subjects. The keys of the first dictionary
        are algorithm, then dataset, then subject. The value of each key is a boolean indicating whether
        the algorithm has been fitted for the subject. Used to avoid unnecesary fits during POMDP simulation.

    results: dict
        Contains the results of all the experiment iterations. An iteration is a combination of different
        analysis variables like data_win_lens, datasets, miss_costs, etc. that are iterated upon. Each
        key of the dictionary represents an sub, which is also a dictionary where each key contains the
        results for an iteration of the experiment. For a dictionary ordered in a 'iterations' -> 'subjects'
        fashion, call the method self.flip_results()

    metadata: dict
        Contains the information about each iteration's parameters for easy access during analysis
    """

    def __init__(
        self,
        epoch_len,
        o_cal_n,
        test_n,
        data_win_lens,
        time_steps,
        datasets,
        algos,
        baseline_methods=None,
        init_delay=0.0,
        time_dependent=False,
        pomdp_go=False,
        downsample=True,
        downsample_target=250,
        add_margins=True,
        mix_values=0.3,
        sarsop_timeout=180,
        hit_reward=10,
        miss_costs=[-100, -1000],
        wait_cost=-1,
        gamma_horizon=0.25,
        solvers="auto",
        simu_modes="data",
        config_path="",
        data_path="",
        solver_path="",
        results_path="",
        models_path="",
        save_models=False,
        load_models=True,
    ):

        self.epoch_len = epoch_len
        self.o_cal_n = o_cal_n
        self.test_n = test_n
        self.data_win_lens = self._check_list(data_win_lens)
        self.time_steps = self._check_list(time_steps)
        self.datasets = self._check_list(datasets)
        self.algos = self._check_list(algos)
        self.baseline_methods = self._check_list(baseline_methods)
        self.init_delay = init_delay
        self.time_dependent = self._check_list(time_dependent)
        self.pomdp_go = self._check_list(pomdp_go)
        self.downsample = downsample
        self.downsample_target = downsample_target
        self.add_margins = add_margins
        self.mix_values = self._check_list(mix_values)
        self.sarsop_timeout = sarsop_timeout
        self.hit_reward = hit_reward
        self.miss_costs = self._check_list(miss_costs)
        self.wait_cost = wait_cost
        self.gamma_horizon = gamma_horizon
        self.solvers = self._check_list(solvers)
        self.simu_modes = self._check_list(simu_modes)
        self.config_path = config_path
        self.data_path = data_path
        self.solver_path = solver_path
        self.results_path = results_path
        self.models_path = models_path
        self.save_models = save_models
        self.load_models = load_models
        self.fitted = self._init_fitted()
        self.params = {}
        self.results = {}
        self.metadata = {}

    @staticmethod
    def _check_list(argument):
        """Check if argument is a list and makes it a list if it isn't"""
        if type(argument) is not list:
            argument = [argument]

        return argument

    def make_sub_list(self, dataset):
        """Get sub list for a given dataset"""
        n_subs = self.params[dataset]["n_subs"]
        excluded_subs = self.params[dataset]["excluded_subs"]

        # MOABB datasets start from 1, while cvep starts from 0
        if dataset == "cvep":
            offset = 0
        else:
            offset = 1

        sub_list = [
            sub_n + offset for sub_n in range(n_subs) if sub_n not in excluded_subs
        ]

        return sub_list

    def _init_fitted(self):
        """Initialization for the fitted attribute"""
        fitted = {}
        for algo in self.algos:
            fitted[algo] = {}
            for dataset in self.datasets:
                fitted[algo][dataset] = {}

        return fitted

    def get_params(self, dataset):
        """Get the configuration params for a specific dataset"""
        with open(
            os.path.join(self.config_path, f"{dataset}.json"), "r"
        ) as dataset_params:
            params = json.loads(dataset_params.read())

        self.params[dataset] = params

    def _load_epochs(self, sub, dataset, ch_keep=None):
        """Different for each experiment"""
        pass

    def _get_tmin_tmax(self):
        """Different for each experiment"""
        pass

    def load_data(self, subject, dataset):
        """Loads data for one subject depending on specific dataset"""
        epochs = self._load_epochs(sub=subject, dataset=dataset)

        labels = epochs.events[..., -1]
        labels -= np.min(labels)

        t_min, t_max = self._get_tmin_tmax(dataset=dataset)
        sfreq = self.params[dataset]["sfreq"]

        data = epochs.get_data()
        data = data[..., t_min:t_max]
        print(
            f"Selecting data from epoch {t_min} to {t_max}, "
            f"corresponding to {t_min / sfreq} and {t_max / sfreq}s"
        )

        return data, labels

    def _get_from_labels(self, labels, dataset):
        """Get label_ids, label_counts and n_class from labels"""
        label_id, label_count = np.unique(labels, return_counts=True)
        n_class = len(label_id)

        self.params[dataset]["label_id"] = label_id
        self.params[dataset]["label_count"] = label_count
        self.params[dataset]["n_class"] = n_class

    def _data_division(self, data, labels, dataset):
        """
        Divide data into cal, cv and test. Different for each experiment.

        Returns
        -------

        X_cal, y_cal: np.array
            Data and labels for calibration of the decoding algorithm

        X_cv, y_cv: np.array
            Data and labels for confusion matrix estimation

        X_test, y_test: np.array
            Data and labels for POMDP simulation
        """
        pass

    def _get_solver(self, time_dependent):
        """
        Logic to determine the solver when 'auto' solver is selected. If the model
        is time-dependent, the solver is set to be SARSOP, otherwise it is set to
        be POMCP
        """
        if time_dependent:
            solver = "SARSOP"
        else:
            solver = "POMCP"

        return solver

    def _load_model(self, model_path):
        """Different for each experiment"""
        pass

    def _save_model(self, model_path, model):
        """
        Save the objects needed for each experiment

        Parameters
        ----------

        model_path: str
            Path to save the model

        model: object
            Object to be saved. Different for each experiment
        """
        # Check if folder exists
        folder_name = os.path.dirname(model_path)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save objects
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

        print()
        print(f"Succesfully saved model to {model_path}")

    @staticmethod
    def get_model_name(pomdp_type, data_win_len, time_step):
        """Different for each experiment"""
        pass

    def fit_clf(self, X_cal, X_cv, y_cal, y_cv, dataset, algo, *args):
        """Different for each experiment"""
        pass

    def get_pomdp_params(self, data_win_len, time_step, time_dependent):
        """
        Get POMDP parameters for a slice length and time step

        Returns
        -------

        pomdp_steps: int
            Number of steps the model will have to simulate the POMDP

        time_steps: list
            List of time steps for the POMDP. Each element of the list is used
            for the initial time of the data window at each POMDP time step

        total_steps: int
            Total number of steps the POMDP will have to simulate

        step_interval: int
            Number of steps to increase in order to get t_max from any given t_min
        """
        if time_dependent:
            # Time steps and step interval. In the time-dependent problem, the data is taken from 0 to an increasing value
            # starting from data_win_len and increasing by time_step for each step (max: self.epoch_len).
            # First we make the end times like so
            end_steps = np.arange(data_win_len, self.epoch_len + 0.01, time_step)

            # If end_steps has a len() of n, we insert n zero values at the beginning, and obtain the start and end times
            # of the epochs at each step by iterating from the beginning and taking i as start and i+n as end. We call
            # this n step_interval
            step_interval = len(end_steps)
            time_steps = np.append(np.zeros((step_interval)), end_steps, 0)

            # This will make the loop throw an IndexError when it tries to take an end time that is outside of the list. This is used
            # during the POMDP simulation to signal the end of the trial
            total_steps = len(time_steps)

            # The total number of POMDP steps should be equal to the number of times we check the conf matrix
            # (total_steps - step_interval) plus 1 (deterministic state transition).
            pomdp_steps = total_steps - step_interval + 1

            # The time-dependent problem is finite horizon, so gamma is equal to or very close to 1
            gamma = 0.99999

        else:
            # Get list for time steps, total steps and step interval
            time_steps = list(
                np.round(np.arange(0, self.epoch_len + 0.01, time_step), 2)
            )
            total_steps = len(time_steps)
            step_interval = int(data_win_len / time_step)

            # Make end_steps for compatibility with td-POMDP
            end_steps = time_steps[step_interval:]  # Unused in regular POMDP

            # The total number of POMDP steps should be equal to the number of times we check the conf matrix
            # plus 1 (deterministic state transition).
            pomdp_steps = total_steps - step_interval + 1

            # Calculate the discount factor as the value that is equal to self.gamma_horizon when elevated
            # to the number of steps the model has, s.t. gamma^pomdp_steps = self.gamma_horizon
            gamma = np.power(self.gamma_horizon, (1 / pomdp_steps))

        return pomdp_steps, end_steps, time_steps, total_steps, step_interval, gamma

    @staticmethod
    def make_iter(pomdp_type, **kwargs):
        """Make iter name based on different parameters"""
        # We don't care about certain parameters if the POMDP is not time-dependent
        excluded_keys = []

        # Remove parameters based on the solver
        if kwargs.get("solver") == "SARSOP":
            excluded_keys.extend(["explorationConst", "initPlanning"])
        elif kwargs.get("solver") == "POMCP":
            excluded_keys.extend(["timeout"])

        iter_name = f"{pomdp_type}POMDP"
        for key, value in kwargs.items():
            if isinstance(value, int):  # Make the cost values positive
                value = np.abs(value)
            if key not in excluded_keys:
                iter_name += f"_{key}-{value}"

        return iter_name

    def conf_matrix_pred(self, clf, X, y, dataset, t_start, t_end, step_n):
        """Different for each experiment"""
        pass

    @staticmethod
    def _add_safety_margin(conf_matrix, mixing_value=0.3):
        """
        Modify a confusion matrix to subtract from the max value of each row and add its zero values,
        in order to account for differences between train and test data. Based on the method described in
        [1].

        References:

            [1] - Park, J., & Kim, K. E. (2012).
                  A POMDP approach to optimizing P300 speller BCI paradigm.
                  IEEE Transactions on Neural Systems and Rehabilitation Engineering, 20(4), 584-594.

        Parameters
        ----------

        conf_matrix: np.array of shape (n_classes, n_classes)
            Confusion matrix to modify

        mixing_value: float, default=0.3
            Parameter q0 in the original formula. The higher the value, the more the matrix is modified.
            A value of 1 would make all the values equal to 1/n_classes, while a value of 0 would leave
            the matrix unchanged.
        """

        copy_matrix = conf_matrix.copy()
        n_class = copy_matrix.shape[0]

        regu_matrix = (1 - mixing_value) * copy_matrix + mixing_value * 1 / n_class

        return regu_matrix

    def normalize_conf_matrix(self, conf_matrix, mix_value, end_steps, time_dependent):
        """
        Normalize confusion matrix according to the POMDP modality
        """
        if time_dependent:
            norm_matrix = np.zeros(conf_matrix.shape)
            n_steps = len(end_steps)

            for step_n in range(n_steps):
                # Get single conf matrix from this time step
                cf = conf_matrix[step_n, ...]

                # Normalize and print
                norm_cf = self._add_safety_margin(cf, mixing_value=mix_value)
                print("")
                print("    Zero values found in the conf matrix, safety margins added")
                print(
                    f"     New confusion matrix (step {step_n + 1}, mix_value {mix_value}):"
                )
                print("\t" + str(norm_cf).replace("\n", "\n\t"))
                print("")

                # Allocate in the norm matrix
                norm_matrix[step_n, ...] = norm_cf
        else:
            # Life is easier for the regular POMDP
            norm_matrix = self._add_safety_margin(conf_matrix, mixing_value=mix_value)
            print("")
            print("    Zero values found in the conf matrix, safety margins added")
            print(f"    New confusion matrix (mix_value {mix_value}):")
            print(norm_matrix)

        return norm_matrix

    def get_conf_matrix(
        self,
        clf,
        X_cv,
        y_cv,
        dataset,
        end_steps,
        time_steps,
        step_interval,
        time_dependent,
        return_cal_score=False,
    ):
        """
        Obtain the POMDP confusion matrix. The method iterates over the POMDP time steps and
        predicts a partial confusion matrix for each. After all predicted labels are obtained,
        the final confusion matrix is calculated.

        Parameters
        ----------

        clf: classifier object
            Classifier to use for the confusion matrix. It is handled by the conf_matrix_pred
            method, that is different for each experiment

        X_cv, y_cv: np.array
            Data and labels for confusion matrix estimation

        dataset: str
            Name of the dataset. Used to get parameters from self.params

        time_steps: list
            List of time steps for the POMDP. Each element of the is used to get the t_min
            of the data window at each POMDP time step

        step_interval: int
            Number of steps to increase the time_steps index in order to get t_max from any
            given t_min

        time_dependent: bool
            If True, the POMDP model is time-dependent. In this case, the confusion matrix is
            a 3D matrix where the first dimension indexes time step

        return_cal_score: bool, default=False
            If True, return the calibration score of the classifier used for the confusion matrix

        Returns
        -------

        conf_matrix: np.array of shape (n_classes, n_classes) or (n_steps, n_classes, n_classes)
            Confusion matrix for the POMDP. If the POMDP model is time-dependent, the confusion matrix
            is a 3D matrix where the first dimension indexes time step. The number of steps corresponds
            to the number of POMDP steps where observations are based on data (i.e. len(end_steps))
        """
        if time_dependent:
            y_pred = np.zeros((len(end_steps), len(y_cv)))
            y_true = np.zeros((len(end_steps), len(y_cv)))
            n_class = len(np.unique(y_cv))

            # We make the conf_matrix per time step when iterating in td-POMDP
            conf_matrix = np.zeros((len(end_steps), n_class, n_class))
        else:
            y_pred = []
            y_true = []

        total_steps = len(time_steps)

        print("")
        for step_n in range(total_steps):
            t_start = time_steps[step_n]
            try:
                t_end = time_steps[step_n + step_interval]
            except IndexError:
                break

            step_pred, step_true = self.conf_matrix_pred(
                clf, X_cv, y_cv, dataset, t_start, t_end, step_n
            )
            partial_score = accuracy_score(step_true, step_pred)
            print(
                f"    Partial score for data from {t_start} to {t_end}: {partial_score}"
            )

            if time_dependent:
                y_pred[step_n, :] = step_pred
                y_true[step_n, :] = step_true

                # Estimate confusion matrix for this time step
                conf_matrix[step_n, :, :] = confusion_matrix(
                    step_true, step_pred, normalize="true"
                )
            else:
                y_pred.extend(step_pred)
                y_true.extend(step_true)

        if time_dependent:
            cal_score = np.mean(
                [
                    accuracy_score(y_true[step_n, :], y_pred[step_n, :])
                    for step_n in range(len(end_steps))
                ]
            )

        else:
            # Estimate the confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
            cal_score = accuracy_score(y_true, y_pred)

        print("")
        print(f"Test data was predicted with an acc of {cal_score}")

        if return_cal_score:
            return conf_matrix, cal_score
        else:
            return conf_matrix

    def _get_all_states(
        self, dataset, pomdp_steps, time_dependent, return_init_states=True
    ):
        """
        Make a list of all the states given a list of labels

        Returns
        -------

        all_states: list of BCIState of TDState objects
            List containing one State object for each state of the problem, whose
            specific class depends on the type of problem

        all_init_states: list of BCIState or TDState objects
            List containing one State object for each possible initial state of the problem
        """
        # Get params
        label_id = self.params[dataset]["label_id"]

        if time_dependent:
            all_states = [
                TDState(int(state), int(time_step))
                for state, time_step in itertools.product(label_id, range(pomdp_steps))
            ]
            all_init_states = [TDState(int(state), 0) for state in label_id]
            all_states.append(TDState("term", 0))

        elif not time_dependent:
            all_states = [BCIState(int(state)) for state in label_id]
            all_init_states = all_states.copy()

        else:
            raise ValueError("The time-dependent parameter is not a boolean????")

        if return_init_states:
            return all_states, all_init_states
        else:
            return all_states

    def _get_init_belief(self, all_states, all_init_states, solver):
        """
        Create initial belief distribution, according to the state space

        Parameters
        ----------

        all_states: list of BCIState or TDState objects
            See self._get_all_states()

        Returns
        -------

        init_belief: pomdp_py.Histogram or pomdp_py.WeightedParticles object
            Initial belief distribution according to the state space and type of problem
        """
        n_init_states = len(all_init_states)
        init_belief = pomdp_py.Histogram(
            {
                state: 1 / n_init_states if state in all_init_states else 0
                for state in all_states
            }
        )
        if solver == "POMCP":
            init_belief = pomdp_py.Particles.from_histogram(init_belief)

        return init_belief

    def _create_problem(
        self,
        init_belief,
        init_true_state,
        n_targets,
        pomdp_steps,
        conf_matrix,
        miss_cost,
        time_dependent,
    ):
        """Make the BCIProblem object depending on the type of problem"""
        if time_dependent:
            problem = TDProblem(
                init_belief,
                init_true_state,
                n_targets,
                n_steps=pomdp_steps,
                conf_matrix=conf_matrix,
                hit_reward=self.hit_reward,
                miss_cost=miss_cost,
                wait_cost=self.wait_cost,
            )

        elif not time_dependent:
            problem = BaseProblem(
                init_belief,
                init_true_state,
                n_targets,
                conf_matrix=conf_matrix,
                hit_reward=self.hit_reward,
                miss_cost=miss_cost,
                wait_cost=self.wait_cost,
            )
        else:
            raise ValueError("The time-dependent parameter is not a boolean????")

        return problem

    def compute_policy(
        self,
        agent,
        gamma,
        solver,
        planning_horizon,
        sarsop_timeout=180,
        planning_time=0.5,
        exploration_const=50,
        policy_path="",
    ):
        """
        Computes the policy for the POMDP

        Parameters
        ----------

        agent: pomdp_py.agent object
            The agent of the POMDP problem, can be accessed fron the created Problem() class
            with Problem.agent

        gamma: int or float
            POMDP's discount factor

        solver: str
            Solver to use for the POMDP. Can be 'POMCP' or 'SARSOP'

        planning_horizon: int
            Number of steps to plan ahead for. Initially equal to pomdp_steps.
            See self.get_pomdp_params() for more info. Note: only used for time-dependent problems

        sarsop_timeout: int, default=180
            Timeout in seconds for SARSOP computation.

        planning_time: float, default=0.5
            Time in seconds to plan for POMCP. Depending on the time-step, the agent has more or less
            time to plan, according to the amount of data it has to wait for before making the next
            obervation.

        exploration_const: int, float or list, default=50
            Exploration constant for POMCP. It determines how the model deals with the exploration-
            exploitation dilemma. A value of 0 makes the model always go towards the actions that
            hold the better value (exploitation), while a higher value inclines the model to visit
            rarely visit nodes (exploration). If it is a list, it will be used for the adaptive
            exploration method. Note: only used for time-dependent problems.

        policy_path: str, default=''
            Full path to policy to use for save/load.
        """
        # Check if there is a saved policy for this iteration
        if self.load_models and os.path.exists(policy_path):
            all_states = list(agent.all_states)
            all_actions = list(agent.all_actions)
            policy = AlphaVectorPolicy.construct(policy_path, all_states, all_actions)

        else:
            if solver == "POMCP":
                policy = POMCP(
                    max_depth=planning_horizon,
                    discount_factor=gamma,
                    planning_time=planning_time,
                    exploration_const=exploration_const,
                    rollout_policy=agent.policy_model,
                )

            elif solver == "SARSOP":
                remove_files = not self.save_models
                policy = sarsop(
                    agent,
                    pomdpsol_path=self.solver_path,
                    discount_factor=gamma,
                    timeout=sarsop_timeout,
                    memory=4096,
                    precision=0.001,
                    remove_generated_files=remove_files,
                    pomdp_name=policy_path.strip(".policy"),
                )
            else:
                raise ValueError(
                    f"The solver parameter is not 'POMCP' or 'SARSOP'. Got: {solver}"
                )

        return policy

    def prepare_trial(self, trial, label, clf, dataset):
        """Make preparations to the trial. Does nothing by default"""
        return trial, label

    @staticmethod
    def print_belief(cur_belief, indent=0):
        """
        Print agent's current belief in a readable manner

        Parameters
        ----------
        cur_belief: pomdp_py 'Particles' or 'Histogram'
            Current belief of the agent. The full class signatures are:
                - Particles: pomdp_py.representations.distribution.particles.Particles
                - Histogram: pomdp_py.representations.distribution.histogram.Histogram

        indent: int, default=0
            Number of whitespaces to add before each printed line. Used for
            consistency with surrounding prints.
        """
        for state in cur_belief:
            b = cur_belief[state]
            if b:  # Particles representation is sparse, we omit zero values
                print(" " * indent + str(state) + " -> " + str(b))

        # elif isinstance(cur_belief, pomdp_py.Histogram):
        #     b = [cur_belief[st] for st in cur_belief]
        #     print(b)

        # else:
        #     raise TypeError("Belief is not Particles or Histogram.")

    def pomdp_pred(self, clf, trial, dataset, t_min, t_max, step_n):
        """Get prediction for the POMDP observation. Varies depending on the experiment"""
        pass

    def update_belief(self, problem, policy, action, observation, solver):
        """Update the POMDP's belief"""
        if solver == "POMCP":
            policy.update(problem.agent, action, observation)  # POMCP belief update
        elif solver == "SARSOP":
            new_belief = pomdp_py.update_histogram_belief(
                problem.agent.cur_belief,
                action,
                observation,
                problem.agent.observation_model,
                problem.agent.transition_model,
                static_transition=False,
            )
            problem.agent.set_belief(
                new_belief
            )  # Belief update on the agent (used for SARSOP)

    def baseline_data_division(self, X_cal, X_cv, X_test, y_cal, y_cv, y_test):
        """Re-arranges data for baseline"""
        pass

    def fit_baesline(self, X_train, X_test, y_train, y_test, dataset, algo):
        """Fits baseline. Different for each experiment"""
        pass

    def run_baseline(self, clf, X_test, y_test, sub, dataset, algo):
        """Runs all baselines and saves scores"""
        pass

    def flip_results(self, keep_old=False):
        """
        Flip the results dictionary from subs -> iterations to iterations -> subs

        Parameters
        ----------

        keep_old: bool, default=False
            If True, make a copy of the old results dictionary and assign it to self.results_old
        """
        if keep_old:
            self.results_old = self.results.copy()

        new_results = defaultdict(dict)
        for sub, iterations in self.results.items():
            for iter_name, metrics in iterations.items():
                new_results[iter_name][sub] = metrics

        self.results = new_results

    def save_results(self):
        """Save results as hdf5 files, containing both results and metadata of each iteration"""
        # Make results path if it doesn't exist
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        for iter_name, scores in self.results.items():
            results_filename = os.path.join(self.results_path, f"{iter_name}.hdf5")
            results_df = pd.DataFrame.from_dict(scores).T

            # Make HDF5 file
            store = pd.HDFStore(results_filename)
            store.put("data", results_df)
            store.get_storer("data").attrs.metadata = self.metadata[iter_name]
            store.close()

    def make_grand_results(self):
        """Print results for each iteration and save iteration averages in a single file"""
        self.grand_results = {}
        metrics = ["FP", "misses", "avg_time", "acc", "n_trials"]
        all_methods = self.baseline_methods.copy().append("POMDP")

        for iter_name, scores in self.results.items():
            # Create shared metrics
            self.grand_results[iter_name] = {f"grand_{metric}": 0 for metric in metrics}
            # Add reward
            self.grand_results[iter_name]["grand_reward"] = 0

            print("")
            print(iter_name)
            print("-" * 30)
            print("")

            # Print and add each sub's metrics to the totals
            for sub in scores.keys():
                print(f"  SUB {sub}:")
                if "baseline" not in iter_name:
                    sub_reward = self.results[iter_name][sub]["total_reward"]
                    print(f"    Total reward (POMDP): {sub_reward}")
                    self.grand_results[iter_name]["grand_reward"] += sub_reward

                for metric in metrics:
                    metric_method = self.results[iter_name][sub][metric]
                    print(f"    {metric}: {metric_method}")
                    self.grand_results[iter_name][f"grand_{metric}"] += metric_method

                print("")

            # Print and save the results for this iteration (dataset, model, etc.)
            print(f"FINAL METRICS ({iter_name})")
            print("-" * 70)
            for metric_name, final_metric in (
                self.grand_results[iter_name].copy().items()
            ):
                if "acc" in metric_name:
                    print("")
                print(f"{metric_name} (all_subjects): {final_metric}")

                if "avgtime" in metric_name:
                    continue
                else:
                    avg_metric = final_metric / len(scores.keys())
                    print(f"Average {metric_name} per subject: {avg_metric}")
                    self.grand_results[iter_name][f"avg_{metric_name}"] = avg_metric

        # Save the comparison results
        grand_results_df = pd.DataFrame(self.grand_results)
        grand_results_df.to_csv(
            os.path.join(self.results_path, "comparison_results.csv")
        )

    def simulate_pomdp(
        self,
        problem,
        policy,
        init_belief,
        pomdp_steps,
        time_steps,
        step_interval,
        mode="data",
        n_trials="auto",
        X_test=None,
        y_test=None,
        clf=None,
        dataset=None,
    ):
        """
        Simulate a POMDP run, either with real data or using the problem's generative model

        Parameters
        ----------

        problem: pomdp_py.POMDP object
            The POMDP problem to be solved, containing the agent and the environment

        policy: POMDP solver object
            The solver to be used to solve the POMDP. Currently supported solvers are SARSOP and POMCP

        init_belief: pomdp_py.Histogram or pomdp_py.Particles object
            The initial belief of the agent. Will be used at the beginning of each trial to rese the belief

        pomdp_steps: int
            Number of POMDP steps to be simulated. In the case of a BCI POMDP problem, this is equal to
            the number of time steps the agent receives observations from the available EEG data, plus a
            last step where the agent can decide with a belief based on the maximum amount of data available

        time_steps: list
            List of time stamps that are used to slice each trial's data in order to obtain observations.
            Only used if mode is 'data'

        step_interval: int
            Number of elements to skip from the initial point in time_steps to reach the corresponding end point

        mode: str, default='data', {'data', 'generative'}
            Simulation mode. If 'data', the agent will receive observations from the available EEG data. If
            'generative', the agent will receive observations from the problem's generative model

        n_trials: str or int, default='auto'
            Number of trials to simulate in generative mode. If set to auto, it will make the number of trials
            based on the provided data

        n_targets: int, default=12
            Number of targets to be simulated. Only used if mode is 'generative'. If mode is 'data', this
            parameter is ignored and the number of targets is inferred from the data

        X_test: np.ndarray, default=None
            EEG data to be used for simulation. Only used if mode is 'data'

        y_test: np.ndarray, default=None
            Labels to be used for simulation. Only used if mode is 'data'

        clf: sklearn classifier object, default=None
            Classifier to be used for simulation. Only used if mode is 'data'

        dataset: str, default=None
            Name of the dataset to be used for simulation. Only used if mode is 'data'

        Returns
        -------

        total_reward: int
            Total reward obtained by the agent during the simulation

        false_positives: int
            Number of false positives obtained by the agent during the simulation. False positives are defined
            as trials where the agent decided to select a target different from the true target

        misses: int
            Number of misses obtained by the agent during the simulation. Misses are defined as trials where
            the agent did not decide by the end of the trial

        total_time: float
            Total time taken by the agent to decide during the simulation across all trials

        beliefs: list of pomdp_py.Histogram objects or pomdp_py.Particles objects
            List of beliefs of the agent at each time step of each trial of the simulation

        n_trials: int
            Number of trials simulated
        """
        # Check correct simulation mode
        if mode not in ["data", "generative"]:
            raise ValueError(
                f"Invalid simulation mode. Must be either data or generative, got: {mode}"
            )

        # Set solver parameter
        if isinstance(policy, AlphaVectorPolicy):
            solver = "SARSOP"
        else:
            solver = "POMCP"

        # Extract parameters from problem
        if isinstance(problem, TDProblem):
            time_dep = True
        else:
            time_dep = False
        miss_cost = problem.agent.reward_model.miss_cost
        conf_matrix = problem.agent.observation_model.observation_matrix
        n_targets = problem.agent.transition_model.n_targets

        if mode == "data":
            n_trials, _, _ = X_test.shape
        elif mode == "generative":
            if n_trials == "auto":
                n_trials, _, _ = X_test.shape

            # Use the same labels than the real data unless the desired number of trials is different
            if n_trials != len(y_test):
                y_test = np.random.randint(n_targets, size=n_trials)

        total_reward = 0
        false_positives = 0
        misses = 0
        total_time = 0.0
        beliefs = []
        total_steps = len(time_steps)

        for trial_n in range(n_trials):
            # Separate next trial and label
            if mode == "data":
                next_trial = X_test[trial_n, :, :]
                next_label = int(y_test[trial_n])
                # Prepare data if needed
                next_trial, next_label = self.prepare_trial(
                    next_trial, next_label, clf, dataset
                )

            else:
                next_label = int(y_test[trial_n])

            # Create the POMDP problem since trials are independent, and set the true state as the next
            # trial
            if time_dep:
                true_state = TDState(next_label, 0)
            else:
                true_state = BCIState(next_label)

            bci_problem = self._create_problem(
                init_belief=init_belief,
                init_true_state=true_state,
                n_targets=n_targets,
                pomdp_steps=pomdp_steps,
                conf_matrix=conf_matrix,
                miss_cost=miss_cost,
                time_dependent=time_dep,
            )

            # Set the true state as the env state
            cur_state = bci_problem.env.state

            print("")
            print(f"TRIAL {trial_n} (true state {cur_state})")
            print("-" * 20)
            data_used = 0  # Tracks how much data has been used as the POMDP progresses
            trial_ended = False  # Flag to break the trial when an action is taken

            # For every time step...
            for step_n in range(total_steps):
                # Adjust step-dependent parameters
                remaining_steps = pomdp_steps - step_n  # Remaining steps for this trial

                print("")
                print(f"  STEP {step_n} ({remaining_steps - 1} steps remaining)")
                print(f"  Current belief (based on {data_used}s of data):")
                cur_belief = bci_problem.agent.cur_belief
                self.print_belief(cur_belief, indent=8)

                # Get your action and execute it
                action = policy.plan(bci_problem.agent)

                # Time-dependent model needs to execute the transition in order to advance
                # the time-step. Normal POMDP did not execute the transiton but the TransitionModel
                # should keep it the same as long as the action is wait. Testing pending for this.
                reward = bci_problem.env.state_transition(action, execute=True)
                cur_state = bci_problem.env.state
                print("")
                print(f"  Action: {action.name}")
                print(f"  Reward: {reward}. Transition to {cur_state}")

                # Add your reward
                total_reward += reward
                if reward == miss_cost:
                    false_positives += 1

                # Check if trial ended (1): Decision taken
                if action.name != "a_wait":
                    decision_time = t_end
                    trial_ended = True
                    end_msg = f"Trial ended with decision {action.name}."

                # Check if trial ended (2): No more data (last time-step)
                # NOTE: Test if this block can become an 'else' for above
                # block. I think it should but I don't want to change anything
                # right now.
                if not trial_ended:
                    t_start = time_steps[step_n]
                    try:
                        t_end = time_steps[step_n + step_interval]
                    except IndexError:
                        decision_time = t_end  # Last step's t_end
                        misses += 1
                        trial_ended = True
                        end_msg = f"Trial ended without a decision."

                # Get observation depending on whether the trial ended or not
                if trial_ended:
                    # End the trial and add the final decision time
                    total_time += decision_time
                    print()
                    print(end_msg)
                    print(f"Decision took {decision_time}s")
                    break
                else:
                    if mode == "data":
                        # Get prediction for observation
                        pred = self.pomdp_pred(
                            clf,
                            next_trial,
                            dataset,
                            t_min=t_start,
                            t_max=t_end,
                            step_n=step_n,
                        )
                        observation = BCIObservation(pred)
                    elif mode == "generative":
                        observation = bci_problem.env.provide_observation(
                            problem.agent.observation_model, action
                        )
                    print(f"  Observation: {observation.name}")
                    # Belief update
                    self.update_belief(
                        bci_problem, policy, action, observation, solver=solver
                    )

        return total_reward, false_positives, misses, beliefs, total_time, n_trials

    def main(self):
        """
        Run the experiment's main loop:
            -> Fit decoder
            -> Compute observation matrix
            -> Solve POMDP
            -> Simulate POMDP
        """
        # First loop: iterate over datasets and algorithms
        for dataset, algo in itertools.product(self.datasets, self.algos):
            # Load dataset-specific parameters
            self.get_params(dataset)

            # Make sub list
            sub_list = self.make_sub_list(dataset)

            # Second loop: iterate over subjects for each combination of algo and dataset
            for sub in sub_list:
                # Make results entry
                self.results[sub] = {}

                # Load data and labels
                data, labels = self.load_data(subject=sub, dataset=dataset)

                # Divide data into calibration, cross-validation (confusion matrix estimation), and test sets
                X_cal, X_cv, X_test, y_cal, y_cv, y_test = self._data_division(
                    data, labels, dataset
                )

                # Fit the decoder for the POMDP (except for TRCA)
                if algo != "trca":
                    clf = self.fit_clf(
                        X_cal, X_cv, y_cal, y_cv, dataset, algo, sub, baseline=False
                    )

                # Third loop: Iterate over all POMDP parameters
                params_iterator = itertools.product(
                    self.data_win_lens,
                    self.time_steps,
                    self.miss_costs,
                    self.mix_values,
                    self.time_dependent,
                    self.pomdp_go,
                    self.solvers,
                    self.simu_modes,
                )

                for iter_parts in params_iterator:
                    # Unpack the iterator
                    (
                        data_win_len,
                        time_step,
                        miss_cost,
                        mix_value,
                        time_dep,
                        pomdp_go,
                        solver,
                        simu_mode,
                    ) = iter_parts

                    # Adjust for automatic solver
                    if solver == "auto":
                        solver = self._get_solver(time_dep)

                    # Get parameters depending on the number of steps for this POMDP, and the discount factor
                    (
                        pomdp_steps,
                        end_steps,
                        time_steps,
                        total_steps,
                        step_interval,
                        gamma,
                    ) = self.get_pomdp_params(data_win_len, time_step, time_dep)

                    print("")
                    print("Creating confusion matrix, POMDP parameters:")
                    print(f"  Time dependent: {time_dep}")
                    print(f"  Model using for simulation: {simu_mode}")
                    print(f"  POMDP-GO: {pomdp_go}")
                    print(f"  Window length: {data_win_len}")
                    print(f"  Time step: {time_step}")
                    if time_dep:
                        print(f"    End of each data window: {end_steps}")
                    print(f"  Cost for misses: {miss_cost}")
                    print(f"  Conf matrix normalization factor: {mix_value}")
                    print(f"  Discount factor: {gamma}")
                    print(f"  Solver: {solver}")
                    if solver == "SARSOP":
                        print(f"    SARSOP solving timeout: {self.sarsop_timeout}")

                    # Create results entry for this iteration
                    if time_dep:
                        pomdp_type = "TD-"
                    else:
                        pomdp_type = ""

                    if pomdp_go:
                        pomdp_type += "GO-"

                    iter_name = self.make_iter(
                        pomdp_type=pomdp_type,
                        dataset=dataset,
                        decoder=algo,
                        winLen=data_win_len,
                        timeStep=time_step,
                        cost=miss_cost,
                        gamma=np.round(gamma, 2),
                        obsSmooth=mix_value,
                        solver=solver,
                        simuMode=simu_mode,
                        timeout=self.sarsop_timeout,
                    )

                    # If the iter name already exists, it means the combination of parameters is redundant and an
                    # equivalent simulation has already been done. This can happen when there some parameters that
                    # are unused in certain scenarios have variations, resulting on serveral identical iterations
                    if iter_name in self.results[sub]:
                        print()
                        print(f"Iteration {iter_name} already exists. Skipping...")
                        continue
                    else:
                        self.results[sub][iter_name] = {}

                    # Get the confusion matrix for this iteration (TRCA fits the clf here)
                    if algo == "trca":
                        clf, conf_matrix, cal_score = self.trca_cv(
                            X_cal,
                            X_cv,
                            y_cal,
                            y_cv,
                            pomdp_type,
                            dataset,
                            algo,
                            sub,
                            iter_name,
                            data_win_len,
                            end_steps,
                            time_steps,
                            step_interval,
                            time_dependent=time_dep,
                        )

                    elif algo != "trca":
                        conf_matrix, cal_score = self.get_conf_matrix(
                            clf,
                            X_cv,
                            y_cv,
                            dataset,
                            end_steps,
                            time_steps,
                            step_interval,
                            time_dep,
                            return_cal_score=True,
                        )

                    self.results[sub][iter_name]["cal_score"] = cal_score

                    # Smooth conf matrix
                    if self.add_margins:
                        # Save the conf_matrix before normalizing
                        self.results[sub][iter_name]["raw_conf_matrix"] = conf_matrix
                        conf_matrix = self.normalize_conf_matrix(
                            conf_matrix, mix_value, end_steps, time_dep
                        )

                    self.results[sub][iter_name]["conf_matrix"] = conf_matrix

                    # POMDP initialization - Different for time-dependent POMDP
                    all_states, all_init_states = self._get_all_states(
                        dataset, pomdp_steps, time_dep, return_init_states=True
                    )
                    init_true_state = random.choice(
                        all_init_states
                    )  # Random initial state

                    # Get initial belief (uniform) - Different for TD-POMDP
                    n_targets = len(all_init_states)
                    init_belief = self._get_init_belief(
                        all_states, all_init_states, solver=solver
                    )

                    bci_problem = self._create_problem(
                        init_belief=init_belief,
                        init_true_state=init_true_state,
                        n_targets=n_targets,
                        pomdp_steps=pomdp_steps,
                        conf_matrix=conf_matrix,
                        miss_cost=miss_cost,
                        time_dependent=time_dep,
                    )

                    print("")
                    print("POMDP instance created!!")
                    # Get policy filename for save_load (use same policy for generative and data modes)
                    policy_name = iter_name.replace("simuMode-generative_", "").replace(
                        "simuMode-data_", ""
                    )
                    policy_path = f"saved_policies/{algo}/{dataset}/sub_{sub}/{policy_name}.policy"

                    # Check if folder exists
                    policy_folder = os.path.dirname(policy_path)
                    if not os.path.exists(policy_folder):
                        os.makedirs(policy_folder)

                    # Create the planner. For regular POMDP (which uses SARSOP), the model is solved
                    # at this step. Time-dependent versions uses POMCP which plans during simulation
                    policy = self.compute_policy(
                        bci_problem.agent,
                        gamma,
                        solver,
                        pomdp_steps,
                        sarsop_timeout=self.sarsop_timeout,
                        policy_path=policy_path,
                    )

                    # Simulate POMDP using the policy
                    print("")
                    print("-" * 70)
                    print(f"POMDP SIMULATION STARTS")
                    print("-" * 70)

                    (
                        total_reward,
                        false_positives,
                        misses,
                        beliefs,
                        total_time,
                        n_trials,
                    ) = self.simulate_pomdp(
                        bci_problem,
                        policy,
                        init_belief,
                        pomdp_steps,
                        time_steps,
                        step_interval,
                        mode=simu_mode,
                        X_test=X_test,
                        y_test=y_test,
                        clf=clf,
                        dataset=dataset,
                    )

                    # Save results for this POMDP
                    self.results[sub][iter_name]["total_reward"] = total_reward
                    self.results[sub][iter_name]["FP"] = false_positives
                    self.results[sub][iter_name]["misses"] = misses
                    self.results[sub][iter_name]["avg_time"] = total_time / (n_trials)
                    self.results[sub][iter_name]["beliefs"] = beliefs
                    self.results[sub][iter_name]["acc"] = (
                        n_trials - (false_positives + misses)
                    ) / n_trials
                    self.results[sub][iter_name]["n_trials"] = n_trials

                    # Store metadata for this POMDP
                    self.metadata[iter_name] = {
                        "template_len": self.epoch_len,
                        "sfreq": self.params[dataset]["sfreq"],
                        "window_len": data_win_len,
                        "time_step": time_step,
                        "gamma": gamma,
                        "hit_reward": self.hit_reward,
                        "miss_cost": miss_cost,
                        "wait_cost": self.wait_cost,
                        "obs_matrix_regu": self.add_margins,
                        "mixing_value": mix_value,
                        "test_n": self.test_n,
                        "cv_n": self.o_cal_n,
                        "algorithm": algo,
                        "solver": solver,
                    }

                # Run baseline
                if self.baseline_methods:
                    X_train, X_test, y_train, y_test = self.baseline_data_division(
                        X_cal, X_cv, X_test, y_cal, y_cv, y_test
                    )
                    baseline_clf = self.fit_baseline(
                        X_train, X_test, y_train, y_test, dataset, algo, sub
                    )
                    self.run_baseline(baseline_clf, X_test, y_test, sub, dataset, algo)

        # Flip self.results: it is saved as subs -> iterations -> metrics, and we want iterations -> subs -> metrics
        self.flip_results(keep_old=True)

        # Save results with metadata as hdf5
        self.save_results()

        # Print iteration results and save comparison results (all subjects avg. for each iteration)
        self.make_grand_results()
        print()
        print(f"Results saved in {self.results_path}. That's all, folks!")

"""
SSVEP-POMDP implementation

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import mne
import os
import pickle

import numpy as np

from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from moabb.datasets import Nakanishi2015

from pomdp_bci.experiment import Experiment
from pomdp_bci.utils import TRCA


class SSVEPExperiment(Experiment):
    """
    SSVEP specific experiment class

    Parameters
    ----------

    init_delay: float, default=0.0
        Initial delay to add to the epochs when slicing the data window that will be used for analyses. This is used
        in VEP-based BCIs, notoriously on TRCA [1], explained as the delay of the human visual pathway to reflect the
        oscilatory activity of the stimuli. According to [1], it should be 0.14 seconds.
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
        baseline_methods,
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

        super().__init__(
            epoch_len,
            o_cal_n,
            test_n,
            data_win_lens,
            time_steps,
            datasets,
            algos,
            baseline_methods,
            init_delay,
            time_dependent=time_dependent,
            pomdp_go=pomdp_go,
            downsample=downsample,
            downsample_target=downsample_target,
            add_margins=add_margins,
            mix_values=mix_values,
            sarsop_timeout=sarsop_timeout,
            hit_reward=hit_reward,
            miss_costs=miss_costs,
            wait_cost=wait_cost,
            gamma_horizon=gamma_horizon,
            solvers=solvers,
            simu_modes=simu_modes,
            config_path=config_path,
            data_path=data_path,
            solver_path=solver_path,
            results_path=results_path,
            models_path=models_path,
            save_models=save_models,
            load_models=load_models,
        )

    def _load_epochs(self, sub, dataset):
        """Load epochs from the dataset for SSVEP. Also save the frequencies in the params dict"""
        if dataset == "nakanishi":
            nakanishi = Nakanishi2015()
            sessions = nakanishi.get_data(subjects=[sub])
            file_path = nakanishi.data_path(sub)

            raw = sessions[sub]["session_0"]["run_0"]
            events = mne.find_events(raw, verbose=False)
            event_id = nakanishi.event_id
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")

        print("")
        print("-" * 70)
        print(f"File path: {file_path}")
        print("-" * 70)

        print("")
        print(f"Loaded data for subject {sub}")

        # Preprocessing
        ch_keep = self.params[dataset]["ch_keep"]
        raw = raw.drop_channels(list(set(raw.ch_names) - set(ch_keep)))

        mne.set_eeg_reference(raw, "average", copy=False, verbose=False)
        print("Dropped unnecesary EEG channels")
        print(f"Channels kept: {raw.ch_names}")

        raw = raw.filter(
            l_freq=3, h_freq=90, method="iir", verbose=False
        )  # Maybe unnecessary due to filterbank
        print("Data was filtered")

        # Epoching
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0,
            tmax=2.2,
            baseline=(0.2, 2.2),
            preload=True,
            verbose=False,
        )
        print("Data was epoched")
        self.params[dataset]["freq_list"] = [
            float(freq) for freq in epochs.event_id.keys() if freq != "boundary"
        ]

        return epochs

    def _get_tmin_tmax(self, dataset):
        sfreq = self.params[dataset]["sfreq"]

        t_min = int(self.init_delay * sfreq) + 1  # In samples
        t_max = int(t_min + self.epoch_len * sfreq)  # In samples

        return t_min, t_max

    def _data_division(self, data, labels, dataset):
        """Divide into cal, cv and test"""
        # Get label_id, label_count and n_class
        self._get_from_labels(labels, dataset)
        label_id = self.params[dataset]["label_id"]
        label_count = self.params[dataset]["label_count"]
        n_class = self.params[dataset]["n_class"]

        print("")
        print(f"Using labels {label_id} with number of trials {label_count}")

        # Number of trials (per class) used for calibration
        a_cal_n = np.unique(label_count)[0] - (self.o_cal_n + self.test_n)
        # Train-test division
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=(n_class * test_n), random_state=42, stratify=labels
        )

        # Divide into cal and cv. In TRCA, these two are united again in order to fit the model
        # using cross-validation
        X_cal, y_cal = X_train[: a_cal_n * n_class], y_train[: a_cal_n * n_class]
        X_cv, y_cv = X_train[a_cal_n * n_class :], y_train[a_cal_n * n_class :]

        return X_cal, X_cv, X_test, y_cal, y_cv, y_test

    @staticmethod
    def get_model_name(pomdp_type, data_win_len, time_step):
        """Get the name of the model based on the parameters"""
        model_name = f"{pomdp_type}POMDP_sliceLen-{data_win_len}_timeStep-{np.round(time_step, 2)}"

        return model_name

    def fit_clf(self, X, y, dataset, algo, sub, t_start, t_end, baseline=False):
        """
        Fit TRCA

        Parameters
        ----------

        t_start: float
            Time to slice the data from, in seconds

        t_end: float
            Time to slice the data to, in seconds

        baseline: bool, default=False
            Whether to fit a baseline TRCA classifier or not. If True, fits a TRCA classifier for each
            baseline epoch length by running this function recursively

        Returns
        -------

        trca: TRCA
            TRCA classifier fitted
        """
        # Get params
        sfreq = self.params[dataset]["sfreq"]
        n_fbands = self.params[dataset]["n_fbands"]
        freq_list = self.params[dataset]["freq_list"]

        if baseline:
            clf_type = "baseline"
        else:
            clf_type = "POMDP"

        if self.downsample:
            downsample = int(sfreq / self.downsample_target)

        trca = TRCA(
            sfreq=sfreq,
            n_fbands=n_fbands,
            peaks=freq_list,
            downsample=self.downsample,
            method="original",
            is_ensemble=True,
        )

        sample_start = int(t_start * sfreq)
        sample_end = int(t_end * sfreq)
        trca.fit(X[..., sample_start:sample_end], y)
        print(
            f"    TRCA ({clf_type}) was fit from {t_start} to {t_end} ({sample_start} to {sample_end})"
        )

        return trca

    def conf_matrix_pred(self, clf, X_cv, y_cv, dataset, t_start, t_end, step_n):
        """
        Predict the confusion matrix for a given time step.

        Parameters
        ----------

        clf: list of TRCA object
            Each elemen of the list is a fit TRCA model that correspond to its index time step

        X_cv: array-like, shape (n_trials, n_channels, n_samples)
            Data to predict on

        y_cv: array-like, shape (n_trials,)
            True labels for the predict data

        dataset: str
            Dataset to use. Used to retrieve parameters from self.params

        t_start: float
            Time to slice the data from, in seconds

        t_end: float
            Time to slice the data to, in seconds

        step_n : int
            Index of the time step to predict on
        """
        # Get parameters
        sfreq = self.params[dataset]["sfreq"]
        model = clf[step_n]

        sample_start = int(t_start * sfreq)
        sample_end = int(t_end * sfreq)
        y_true = list(y_cv)

        # Predict
        y_pred = model.predict(X_cv[..., sample_start:sample_end])

        return y_pred, y_true

    def _load_model(self, model_path):
        """
        Load a model from a given path

        Parameters
        ----------

        model_path: str
            Path to the model to load

        Returns
        -------

        best_model: object
            Model to load, saved with pickle

        best_matrix: array-like, shape (n_class, n_class) or (n_steps, n_class, n_class)
            Confusion matrix for the best model

        best_score: float
            Score for the best model
        """
        with open(model_path, "rb") as model_file:
            best_model, best_matrix, best_score = pickle.load(model_file)

        print(f"    Successfully loaded model {os.path.basename(model_path)}")

        return best_model, best_matrix, best_score

    def trca_cv(
        self,
        X_cal,
        X_cv,
        y_cal,
        y_cv,
        model_type,
        dataset,
        algo,
        sub,
        iter_name,
        slice_len,
        end_steps,
        time_steps,
        step_interval,
        time_dependent,
    ):
        """
        TRCA cross-validation process for POMDP. Since the data that is usually
        used to train TRCA needs to be split between TRCA calibration and POMDP
        confusion matrix estimation (cal and cv in the context of this class),
        this allows to run a cross-validation process for TRCA in order to find
        the combination that produces the best results.

        This process includes the following steps:
            1. Create a cross-validation iterator
            2. For each cross-validation fold:
                2.1 Fit a TRCA model for each POMDP time-step, and predict on the
                    corresponding cv set
                2.2 Calculate partial accuracy for each time-step and add the
                    predicted values to a general prediction array
                2.3 Calculate the general accuracy of the cross-validation fold
                    and update the best fold if necessary
            3. Take the collection of models (one per time step) that produce
               the best general score and confusion matrix and save them
               in a dictionary indexed by the POMDP iteration name

        Parameters
        ----------

        X_cal : array-like, shape (n_trials, n_channels, n_samples)
            Calibration data. Will be joined with cv data in order to split using
            cross-validation iterators

        X_cv : array-like, shape (n_trials, n_channels, n_samples)
            Cross-validation data. Will be joined with cal data in order to split using
            cross-validation iterators

        y_cal : array-like, shape (n_trials,)
            Calibration labels. Will be joined with cv labels in order to split using
            cross-validation iterators

        y_cv : array-like, shape (n_trials,)
            Cross-validation labels. Will be joined with cal labels in order to split using
            cross-validation iterators

        model_type: str
            Type of model to use. Can be 'TD-' or ''

        dataset: str
            Dataset name. Used to get the parameters from self.params

        algo: str
            Decoder name.

        sub: str
            Subject name. Used to get set parameters in self.results

        iter_name: str
            Name of the POMDP iteration. Used to index the results in self.results

        slice_len: float
            Length of the data epochs in seconds

        end_steps: list
            List of the time steps equal to the max length of data for each POMDP time step. Only used
            if self.time_dependent is True. Check Experiment.get_pomdp_params for more information

        time_steps: list
            List of time steps to be used in the POMDP iteration. Check Experiment.get_pomdp_params
            for more information

        step_interval: int
            Number of steps to increase the index of time_steps from any t_min to get the corresponding
            t_max. Check Experiment.get_pomdp_params for more information

        time_dependent: bool
            Whether the POMDP iteration is time-dependent or not

        Returns
        -------

        best_model : list
            Dictionary containing the best model for each time-step. Each element corresponding to
            the TRCA model for that POMDP time step.

        best_matrix: array-like, shape (n_class, n_class) or (n_steps, n_class, n_class)
            Confusion matrix corresponding to best_model. The conf matrix is obtained by accumulating
            the predictions for each POMDP time_step
        """
        # Check if there is a saved model for this iteration
        time_step = np.diff(end_steps)[0]
        model_name = self.get_model_name(model_type, slice_len, time_step)
        model_filename = os.path.join(
            self.models_path, f"{algo}/{dataset}/sub_{sub}/{model_name}.pkl"
        )

        if self.load_models and os.path.exists(model_filename):
            print()
            print(f"  A model for this iteration already exists. Loading...")
            best_model, best_matrix, best_score = self._load_model(model_filename)
        else:
            # Join cal and cv data and labels
            X_train = np.concatenate((X_cal, X_cv), axis=0)
            y_train = np.concatenate((y_cal, y_cv), axis=0)

            # CV splitter
            n_class = self.params[dataset]["n_class"]
            cv_splitter = StratifiedShuffleSplit(
                n_splits=12, test_size=(n_class * self.o_cal_n), random_state=42
            )

            # Run cross-validation and keep the clf that produces the best confusion matrix
            print("")
            print(f"Creating confusion matrix using an epoch length of {slice_len}")
            best_score = 0
            best_split = 0
            best_matrix = 0

            for split_n, (train_idx, test_idx) in enumerate(
                cv_splitter.split(X_train, y_train)
            ):
                X_cal, X_cv = X_train[train_idx], X_train[test_idx]
                y_cal, y_cv = y_train[train_idx], y_train[test_idx]

                # Metrics
                print("")
                print(f"Split {split_n}")
                n_steps = len(time_steps)  # Same as total_steps
                models = {}  # Fit one TRCA model per time step

                for step_n in range(n_steps):
                    t_start = time_steps[step_n]
                    try:
                        t_end = time_steps[step_n + step_interval]
                    except IndexError:
                        continue

                    # Fit TRCA model
                    trca = self.fit_clf(
                        X_cal, y_cal, dataset, algo, sub, t_start, t_end, baseline=False
                    )
                    models[step_n] = trca

                # Get conf matrix for this split
                conf_matrix, score = self.get_conf_matrix(
                    models,
                    X_cv,
                    y_cv,
                    dataset,
                    end_steps,
                    time_steps,
                    step_interval,
                    time_dependent,
                    return_cal_score=True,
                )

                if score >= best_score:
                    best_score = score
                    best_split = split_n + 1
                    best_model = models
                    best_matrix = conf_matrix

        print("")
        print(f"Best score: {best_score}")

        if time_dependent:
            print("Best confusion matrices")
            for step_n in range(best_matrix.shape[0]):
                cf = best_matrix[step_n]
                print(f"Confusion matrix (step {step_n + 1}): ")
                print("\t" + str(cf).replace("\n", "\n\t"))
        else:
            print("Best confusion matrix:")
            print(best_matrix)

        # Save model
        if self.save_models:
            to_save = (best_model, best_matrix, best_score)
            self._save_model(model_filename, to_save)

        return best_model, best_matrix, best_score

    def pomdp_pred(self, clf, trial, dataset, t_min, t_max, step_n):
        """Predict using the TRCA model to get a POMDP observation"""
        # Get params
        sfreq = self.params[dataset]["sfreq"]

        # Get prediction from the corresponding model
        sample_start = int(t_min * sfreq)
        sample_end = int(t_max * sfreq)

        pred_label = clf[step_n].predict(trial[..., sample_start:sample_end])[0]
        pred = int(pred_label)

        return pred

    @staticmethod
    def baseline_data_division(X_cal, X_cv, X_test, y_cal, y_cv, y_test):
        """Re-arrange data into train and test sets for baseline analysis"""
        # Join cal and cv data and labels
        X_train = np.concatenate((X_cal, X_cv), axis=0)
        y_train = np.concatenate((y_cal, y_cv), axis=0)

        return X_train, X_test, y_train, y_test

    def fit_baseline(self, X_train, X_test, y_train, y_test, dataset, algo, sub):
        """
        Fits models for baseline analysis. In the case of TRCA, this requires a separate
        model for each baseline method (which correspond to TRCA at different epoch lens)

        Returns
        -------

        baseline_clf : dict of TRCA models
            Dictionary containing the TRCA models for each baseline method
        """
        baseline_clf = {}

        for baseline in self.baseline_methods:
            # Get t_start and t_end
            t_start = 0
            t_end = float(baseline.split("_")[-1])

            # Fit clf
            baseline_clf[baseline] = self.fit_clf(
                X_train, y_train, dataset, algo, sub, t_start, t_end, baseline=True
            )

        return baseline_clf

    def run_baseline(self, clf, X_test, y_test, sub, dataset, algo):
        """
        Run SSVEP baseline analysis
        """
        # Get params
        sfreq = self.params[dataset]["sfreq"]
        n_trials, _, _ = X_test.shape

        for baseline_method in self.baseline_methods:
            iter_name = f"baseline_{baseline_method}"
            self.results[sub][iter_name] = {}

            # Get t_end
            t_end = float(baseline_method.split("_")[-1])
            sample_end = int(t_end * sfreq)

            # Get predictions
            labels_pred = clf[baseline_method].predict(X_test[..., :sample_end])
            mean_time = t_end

            # Get acc and FP
            ssvep_acc = accuracy_score(y_test, labels_pred)
            ssvep_fp = n_trials - np.sum(labels_pred == y_test)
            ssvep_avg_time = mean_time

            # Save results
            self.results[sub][iter_name]["acc"] = ssvep_acc
            self.results[sub][iter_name]["misses"] = 0
            self.results[sub][iter_name]["FP"] = ssvep_fp
            self.results[sub][iter_name]["avg_time"] = ssvep_avg_time
            self.results[sub][iter_name]["n_trials"] = n_trials

            # Make metadata
            self.metadata[iter_name] = {
                "template_len": self.epoch_len,
                "sfreq": sfreq,
                "window_len": "",
                "time_step": "",
                "gamma": "",
                "hit_reward": "",
                "miss_cost": "",
                "wait_cost": "",
                "obs_matrix_regul": "",
                "mixing_value": "",
                "mixing_method": "",
                "mix_step": "",
                "min_mix": "",
                "test_n": self.test_n,
                "cv_n": "",
                "algorithm": algo,
                "method": baseline_method,
            }


if __name__ == "__main__":
    time_dependent = [True, False]
    simu_modes = ["data", "generative"]
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_path = os.path.join(os.getcwd(), f"results/SSVEP_refactortest_{time}")

    # Path variables - put your paths here or export them to your environment
    data_path = os.environ.get("SSVEP")
    sarsop_path = os.environ.get("SARSOP")
    models_path = os.environ.get("MODELS")

    # Data variables
    init_delay = 0.13  # In seconds, 0.13 in the original TRCA paper
    epoch_len = 1.0
    test_n = 2  # Number of trials (per class) used for POMDP simulation
    cv_n = 5  # Number of trials (per class) used for Confusion matrix calculation
    mixing_values = [0.3, 0.01]
    gamma_horizon = 0.25

    # Analysis variables
    data_win_lens = [0.5]
    time_steps = [0.1]
    datasets = ["nakanishi"]
    algos = ["trca"]
    baseline_methods = ["trca_0.5", "trca_1"]

    exp = SSVEPExperiment(
        epoch_len=epoch_len,
        o_cal_n=cv_n,
        test_n=test_n,
        data_win_lens=data_win_lens,
        time_steps=time_steps,
        datasets=datasets,
        algos=algos,
        baseline_methods=baseline_methods,
        init_delay=init_delay,
        time_dependent=time_dependent,
        mix_values=mixing_values,
        solvers="SARSOP",
        simu_modes=simu_modes,
        config_path="config",
        data_path=data_path,
        solver_path=sarsop_path,
        results_path=results_path,
        models_path=models_path,
        save_models=True,
        load_models=True,
        gamma_horizon=gamma_horizon,
    )

    exp.main()

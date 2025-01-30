"""
CVEP-POMDP implementation

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import keras
import mne
import os
import pickle

import numpy as np

from collections import OrderedDict
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

from bci_pomdp.experiment import Experiment
from bci_pomdp.utils import vanilliaEEG2Code, EEGnet_patchembeddingdilation


class CVEPExperiment(Experiment):
    """
    CVEP specific experiment class

    Parameters
    ----------


    code_win_len: float, default=0.250
        Size (in seconds) of the sliding window used to regress the code out of EEG activity for code-VEP BCI.

    Attributes
    ----------

    codes: dict
        Structure containing the codes for each of the BCI targets, together with their labels. These are extracted
        from each participant's event dictionary.
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
        code_win_len=0.250,
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

        self.code_win_len = code_win_len

    def _load_epochs(self, sub, dataset):
        """Loads data for one subject depending on specific dataset"""

        # CVEP needs annotation cleaning before extracting events
        if dataset == "cvep":
            filename = f"P{sub}_whitemseq.set"

            file_path = os.path.join(self.data_path, filename)
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

            for idx in range(len(raw.annotations.description)):
                code = raw.annotations.description[idx].split("_")[0]
                lab = raw.annotations.description[idx].split("_")[1]
                code = code.replace("\n", "")
                code = code.replace("[", "")
                code = code.replace("]", "")
                code = code.replace(" ", "")
                raw.annotations.description[idx] = code + "_" + lab

            events, event_id = mne.events_from_annotations(raw, verbose=False)
        else:
            raise NotImplementedError("Dataset {dataset} not implemented")

        print("")
        print("-" * 70)
        print(f"File path: {file_path}")
        print("-" * 70)

        print("")
        print(f"Loaded data for subject {sub}")

        # Preprocessing
        ch_keep = self.params[dataset]["ch_slice"]
        raw = raw.drop_channels(list(set(raw.ch_names) - set(ch_keep)))

        # CVEP needs the montage manually set
        if dataset == "cvep":
            raw = raw.drop_channels(["21", "10"])

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

        # Analysis
        codes = OrderedDict()  # Make an ordered dict with the code for each class
        for k, v in epochs.event_id.items():
            code = k.split("_")[0]
            idx = k.split("_")[1]
            codes[v - 1] = np.array(list(map(int, code)))

        # Assign dataset parameters
        self.params[dataset]["codes_true"] = codes

        return epochs

    def _get_tmin_tmax(self, dataset):
        sfreq = self.params[dataset]["sfreq"]

        t_min = int(self.init_delay * sfreq) + 1  # In samples
        t_max = int(
            t_min + (self.epoch_len + self.code_win_len) * sfreq
        )  # Also in samples

        return t_min, t_max

    def _data_division(self, data, labels, dataset):
        """Divide data into cal, cv and test. Also set the codes for later use"""
        # Get label_id, label_count and n_class
        self._get_from_labels(labels, dataset)
        label_id = self.params[dataset]["label_id"]
        label_count = self.params[dataset]["label_count"]
        n_class = self.params[dataset]["n_class"]

        print("")
        print(f"Using labels {label_id} with number of trials {label_count}")

        # Divide data in train and test (for baseline), and further divide train data into calibration and cv
        # (for POMDP training and conf matrix creation)
        a_cal_n = np.unique(label_count)[0] - (
            self.o_cal_n + self.test_n
        )  # Number of trials (per class) used for calibration
        train_n = (
            a_cal_n + self.o_cal_n
        )  # Number of trials (per class) used for calibration + conf_matrix generation

        X_train, y_train = (
            data[: train_n * n_class],
            labels[: train_n * n_class],
        )  # The first train_n trials
        X_test, y_test = data[train_n * n_class :], labels[train_n * n_class :]

        # X_cal is used to train the NN that the POMDP uses, and X_cv to calculate the confusion matrix
        X_cal, y_cal = X_train[: a_cal_n * n_class], y_train[: a_cal_n * n_class]
        X_cv, y_cv = X_train[a_cal_n * n_class :], y_train[a_cal_n * n_class :]

        return X_cal, X_cv, X_test, y_cal, y_cv, y_test

    def _epoch_to_window(
        self,
        data,
        labels,
        dataset,
        win_format="channels_last",
        refresh=60,
        return_code_len=False,
    ):
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
            If True, return the length of the code per trial.

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
        # Get params
        sfreq = self.params[dataset]["sfreq"]
        codes = self.params[dataset]["codes_true"]

        # Handle single trial by inserting new dim at axis 0
        if data.ndim == 2:
            data = np.expand_dims(data, 0)

        # Get useful parameters from the data
        n_trials, n_channels, n_samples = data.shape
        win_samples = int(self.code_win_len * sfreq)
        code_len = int(n_samples - win_samples)

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
                win_data[sample_count] = trial[:, idx : idx + win_samples]
                # If the current sample is over the duration of the current bit, increase the bit count
                if idx / sfreq >= (code_pos + 1) / refresh:
                    code_pos += 1
                # Assign label to the sample according to the corresponding code bit
                win_labels[sample_count] = int(code[code_pos])
                sample_count += 1

        # Insert new dimension to the data
        if win_format == "channels_first":
            dim_pos = 1
        elif win_format == "channels_last":
            dim_pos = -1
        else:
            warnings.warn(
                "Invalid win_format parameter, using 'channels_last as default...",
                UserWarning,
            )
            dim_pos = -1

        win_data = np.expand_dims(win_data, dim_pos)
        win_data = win_data.astype(np.float32)  # Format as float

        # One-hot encoding of labels for the NN
        win_labels = np.vstack((win_labels, np.abs(1 - win_labels))).T

        if return_code_len:
            return win_data, win_labels, code_len
        else:
            return win_data, win_labels

    @staticmethod
    def predict_codes(clf, data_test, n_class):
        """Predict the codes on data_test"""
        test_codes_pred = []
        bits_per_class = int(len(data_test) / n_class)
        pred_splits = 11
        for i in range(pred_splits):
            if i == pred_splits - 1:
                data_slice = data_test[bits_per_class * i :]
            else:
                data_slice = data_test[bits_per_class * i : bits_per_class * (i + 1)]

            pred = clf.predict(data_slice, batch_size=64)
            test_codes_pred.extend(pred[:, 0])

        codes_pred = np.array(test_codes_pred)  # In window format

        return codes_pred

    @staticmethod
    def _load_model(model_path, algo):
        """
        Load a model from the path

        Parameters
        ----------

        model_path: str
            Path to the model to be loaded

        Returns
        -------

        clf: keras.model
            Loaded model

        train_std: float
            Standard deviation of the training data

        codes_pred: np.array
            Predicted codes for the test data
        """
        if algo in ["EEGnet_patchembeddingdilation"]:
            keras_filename = model_path.strip(".pkl") + ".h5"
            clf = keras.models.load_model(keras_filename)

            with open(model_path, "rb") as model_file:
                train_std, codes_pred, code_len = pickle.load(model_file)

        else:
            with open(model_path, "rb") as model_file:
                clf, train_std, codes_pred, code_len = pickle.load(model_file)

        print(f"    Successfully loaded model {os.path.basename(model_path)}")

        return clf, train_std, codes_pred, code_len

    @staticmethod
    def _save_model(model_path, algo, model):
        """
        Save the objects needed for the model

        Parameters
        ----------

        model_path: str
            Path to the model to be saved

        algo: str
            Name of the training algorithm used (affects how the model is saved)

        model: tuple
            Tuple containing the objects to be saved
        """
        # Handle saving for keras models
        if algo in ["EEGnet_patchembeddingdilation"]:
            keras_model, train_std, codes_pred, code_len = model
            model = (train_std, codes_pred, code_len)

            keras_filename = model_path.strip(".pkl") + ".h5"
            keras_model.save(keras_filename)

        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

        print()
        print(f"Successfully saved model to {model_path}")

    def fit_clf(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        dataset,
        algo,
        sub,
        baseline=False,
        *args,
    ):
        """
        Fit the selected algorithm. In the case of CVEP, also predict on the
        codes of the test data and assign them as a property of the class
        for easy access later.

        Parameters
        ----------

        algo: str
            Name of the training algorithm to be used

        baseline: bool, default=False
            Whether the clf is for baseline or not. Used to separate the std of the training
            data that gets stored for later use

        Returns
        -------

        clf: clf object
            Clf object that has already been fit
        """
        print("")
        print("Fitting data for POMDP...")
        n_class = self.params[dataset]["n_class"]

        # Get clf_type to separate std and codes pred
        if baseline:
            clf_type = "baseline"
        else:
            clf_type = "POMDP"

        # Check if there is a saved model for this iteration
        model_name = f"{clf_type}_model"
        model_filename = os.path.join(
            self.models_path, f"{algo}/{dataset}/sub_{sub}/{model_name}.pkl"
        )

        if self.load_models and os.path.exists(model_filename):
            print()
            print(f"  A model for this iteration already exists, loading...")
            clf, train_std, codes_pred, code_len = self._load_model(
                model_filename, algo
            )

        else:
            # Pass calibration and CV data to window format
            data_train, labels_train, code_len = self._epoch_to_window(
                X_train, y_train, dataset, return_code_len=True
            )
            data_test, labels_test = self._epoch_to_window(X_test, y_test, dataset)

            # Data needs to be normalized
            train_std = data_train.std(axis=0)
            data_train /= train_std + 1e-8
            data_test /= train_std + 1e-8

            # Class balancing: There are more 1s than 0s in our training codes, so we use a random
            # under sampler to make it balanced
            rus = RandomUnderSampler()
            counter = np.array(range(0, len(labels_train))).reshape(-1, 1)
            index, _ = rus.fit_resample(counter, labels_train[:, 0])
            data_train = np.squeeze(data_train[index, :, :, :], axis=1)
            labels_train = np.squeeze(labels_train[index])

            # Initialize NN
            win_samples = int(self.code_win_len * self.params[dataset]["sfreq"])
            n_channels = data_train.shape[
                1
            ]  # Number of channels in the data (for channels last)

            if algo == "vanilliaEEG2Code":
                clf = vanilliaEEG2Code(
                    windows_size=win_samples, n_channel_input=n_channels
                )
            elif algo == "EEGnet_patchembeddingdilation":
                clf = EEGnet_patchembeddingdilation(
                    windows_size=win_samples, n_channel_input=n_channels
                )
            else:
                raise NameError("Incorrect algo name")

            print("")
            clf.summary()

            # Select optimizer and compile classifier for fit
            batchsize = 256
            nn_epochs = 45
            learning_rate = 1e-3
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
            clf.compile(
                loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
            )
            print("")
            print("Classifier compiled. Fitting...")
            print("")
            history = clf.fit(
                data_train,
                labels_train,
                batch_size=batchsize,
                epochs=nn_epochs,
                shuffle=True,
                verbose=0,
            )
            keras.backend.clear_session()
            print("Data was fit")

            # Predict the codes of validation data (split in 10 to avoid OOM from the GPU)
            codes_pred = self.predict_codes(clf, data_test, n_class=n_class)

        # Save the codes pred and the code_len to be retrieved later
        self.params[dataset][clf_type + "-codes_pred"] = codes_pred
        self.params[dataset]["code_len"] = code_len

        # Needs to be used again for normalizing the data later
        self.params[dataset][clf_type + "-train_std"] = train_std

        # Save the model
        if self.save_models:
            to_save = (clf, train_std, codes_pred, code_len)
            self._save_model(model_filename, algo, to_save)

        return clf

    @staticmethod
    def get_code_prediction(
        codes_true,
        codes_pred,
        sfreq,
        code_len,
        t_min=0.0,
        t_max=None,
        refresh=60,
        incremental=False,
    ):
        """
        Make predictions based on regressed codes

        Pearameters
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
        n_trials = int(len(codes_pred) / code_len)

        # Final labels list
        labels_pred = []
        mean_time = []

        for trial in range(n_trials):
            # Get the next trial
            trial_code = codes_pred[trial * code_len : (trial + 1) * code_len]

            # Transform the code from EEG sampling rate to screen refresh rate by averaging predictions
            code_buffer = []
            code_pos = 0
            y_tmp = []
            for idx in range(len(trial_code)):
                y_tmp.append(trial_code[idx])
                if idx / sfreq >= (code_pos + 1) / refresh:
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
                points = [
                    (sample_start, end) for end in range(sample_end, max_sample + 1, 3)
                ]

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
                    if (
                        (corrs[corrs_idx[-1]] - corrs[corrs_idx[-2]])
                        / corrs[corrs_idx[-1]]
                        > 0.5
                    ) and (p_max < 1e-3):
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

    def conf_matrix_pred(self, clf, X_cv, y_cv, dataset, t_start, t_end, step_n):
        """
        Predict on data used to estimate the confusion matrix

        Parameters
        ----------

        clf: clf object
            Unused for CVEP, here for compliance with other experiments

        X_cv: np.array, shape (n_trials, n_channels, n_samples)
            Data to predict on. NOTE: Unused in CVEP, as the codes_pred are stored
            upon clf fit.

        y_cv: np.array, shape (n_trials,)
            True labels of the data to predict on

        Returns
        -------

        y_pred: np.array
            Predicted labels for each trial of the o-cal data (cv data)

        y_true: np.array
            True labels for each trial of the o-cal data (cv data)
        """
        # Get params
        sfreq = self.params[dataset]["sfreq"]
        codes_true = self.params[dataset]["codes_true"]
        codes_pred = self.params[dataset]["POMDP-codes_pred"]
        code_len = self.params[dataset]["code_len"]

        win_samples = int(self.code_win_len * sfreq)
        y_true = list(y_cv)  # Labels are the same no matter the slice

        # Predict data on current time step and add to pred array. This goes back to non-window format
        y_pred = self.get_code_prediction(
            codes_true=codes_true,
            codes_pred=codes_pred,
            sfreq=sfreq,
            code_len=code_len,
            t_min=t_start,
            t_max=t_end,
        )

        return y_pred, y_true

    def prepare_trial(self, trial, label, clf, dataset):
        """
        Prepare trial. In the CVEP experiment, this involves:
            - Convert trial data to window format
            - Get code prediction
            - Return predicted code

        Note that the label is unchanged.
        """
        # Convert them to window and normalize
        std = self.params[dataset]["POMDP-train_std"]
        win_data, _ = self._epoch_to_window(trial, label, dataset)
        win_data /= std + 1e-8

        # Predict the code (take the first column only)
        code_pred = clf.predict(win_data, batch_size=64)[:, 0]

        return code_pred, label

    def pomdp_pred(self, clf, trial, dataset, t_min, t_max, step_n):
        """
        Get POMDP prediction.

        Parameters
        ----------

        clf: clf object
            CNN that regresses the codes. Unused in CVEP, here for compatibility with the
            abstract class

        trial: np.array
            Trial to classified. In CVEP it is the predicted code from the EEG activity

        step_n: int
            Only used for compatibility with the abstract class and other experiments

        Rrturns
        -------

        pred: int
            Predicted label for the input trial
        """
        codes_true = self.params[dataset]["codes_true"]
        sfreq = self.params[dataset]["sfreq"]
        code_len = self.params[dataset]["code_len"]

        preds = self.get_code_prediction(
            codes_true=codes_true,
            codes_pred=trial,
            code_len=code_len,
            sfreq=sfreq,
            t_min=t_min,
            t_max=t_max,
        )

        pred = int(preds[0])
        return pred

    def baseline_data_division(self, X_cal, X_cv, X_test, y_cal, y_cv, y_test):
        """
        Re-arrange data for baseline experiments. In the case of CVEP, calibration and
        cv data are concatenated, as they are divided without shuffling.

        Returns
        -------

        X_train: np.array of shape (n_trials, n_channels, n_samples)
            Data corresponding to X_cal and X_cv, concatenated on the first axis

        X_test: np.array of shape (n_trials, n_channels, n_samples)
            Test data corresponding to X_test. Untouched in CVEP

        y_train: np.array of shape (n_trials,)
            Labels corresponding to X_train, concatenating those for y_cal and y_cv

        y_test: np.array of shape (n_trials,)
            Labels corresponding to X_test. Untouched in CVEP
        """
        X_train = np.concatenate((X_cal, X_cv), axis=0)
        y_train = np.concatenate((y_cal, y_cv), axis=0)

        return X_train, X_test, y_train, y_test

    def fit_baseline(self, X_train, X_test, y_train, y_test, dataset, algo, sub):
        """
        Fit baseline clf for CVEP. Since CVEP uses the same NN for all baseline methods,
        this requires to call the fit_clf method with the corresponding data.
        """
        baseline_clf = self.fit_clf(
            X_train, X_test, y_train, y_test, dataset, algo, sub, baseline=True
        )

        return baseline_clf

    @staticmethod
    def make_preds_cumul_aggresive(
        codes_true, codes_pred, sfreq, code_len, min_len=30, refresh=60
    ):
        """
        Always make a prediction and it's the template that correlate the most
        during the last step of window growth.
        Stop the process if the same prediction is made 40 times in the row
        and pick this prediction.
        """

        codes_pred = np.array(codes_pred)
        n_trials = int(len(codes_pred) / code_len)

        # Final lists
        labels_pred = []
        mean_time = []

        for trial in range(n_trials):
            # Retrieve a trial
            trial_code = codes_pred[trial * code_len : (trial + 1) * code_len]
            code_pos = 0

            # Do an average over the prdata, codes, labels, sfreq
            code_buffer = []
            y_tmp = []
            for idx in range(len(trial_code)):
                y_tmp.append(trial_code[idx])
                if idx / sfreq >= (code_pos + 1) / refresh:
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
                    dtw_values.append(
                        np.corrcoef(code_buffer[:long], values[:long])[0, 1]
                    )
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

    def run_baseline(self, clf, X_test, y_test, sub, dataset, algo):
        """
        Run CVEP baseline classification.
        """
        # Get params
        codes_true = self.params[dataset]["codes_true"]
        codes_pred = self.params[dataset]["baseline-codes_pred"]
        sfreq = self.params[dataset]["sfreq"]
        code_len = self.params[dataset]["code_len"]
        n_trials, _, _ = X_test.shape

        for clf_method in self.baseline_methods:
            iter_name = f"baseline_{clf_method}"
            self.results[sub][iter_name] = {}

            if clf_method == "cvep_pvalue":
                labels_pred, mean_time = self.get_code_prediction(
                    codes_true=codes_true,
                    codes_pred=codes_pred,
                    sfreq=sfreq,
                    code_len=code_len,
                    t_min=0.0,
                    t_max=0.5,
                    incremental=True,
                )
            elif clf_method == "cvep_cumul":
                labels_pred, mean_time = self.make_preds_cumul_aggresive(
                    codes_true=codes_true,
                    codes_pred=codes_pred,
                    sfreq=sfreq,
                    code_len=code_len,
                )
            elif clf_method == "cvep_static":
                labels_pred = self.get_code_prediction(
                    codes_true=codes_true,
                    codes_pred=codes_pred,
                    sfreq=sfreq,
                    code_len=code_len,
                    t_min=0.0,
                    t_max=0.5,
                    incremental=False,
                )
                mean_time = 0.5
            else:
                raise NameError("Invalid baseline method")

            # Get acc, FPs, misses, etc.
            acc = accuracy_score(y_test, labels_pred)
            misses = np.count_nonzero(labels_pred == -1)
            fp = (
                n_trials - misses - np.sum(y_test == labels_pred)
            )  # n_trials - misses - true positives
            avg_time = mean_time

            # Save results
            self.results[sub][iter_name]["acc"] = acc
            self.results[sub][iter_name]["misses"] = misses
            self.results[sub][iter_name]["FP"] = fp
            self.results[sub][iter_name]["avg_time"] = avg_time
            self.results[sub][iter_name]["n_trials"] = n_trials

            # Make metadata
            self.metadata[iter_name] = {
                "template_len": self.epoch_len + self.code_win_len,
                "sfreq": sfreq,
                "window_len": "",
                "time_step": "",
                "gamma": "",
                "hit_reward": "",
                "miss_cost": "",
                "wait_cost": "",
                "obs_matrix_regu": "",
                "mixing_value": "",
                "mixing_method": "",
                "mix_step": "",
                "min_mix": "",
                "test_n": self.test_n,
                "cv_n": "",
                "algorithm": algo,
                "method": clf_method,
            }


if __name__ == "__main__":
    time_dependent = [True, False]
    simu_modes = ["data", "generative"]
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_path = os.path.join(os.getcwd(), f"results/CVEP_refactortest_{time}")

    # Path variables
    data_path = os.environ["CVEP"]
    sarsop_path = os.environ["SARSOP"]
    models_path = os.environ["MODELS"]

    # Data variables
    epoch_len = 1.01
    test_n = 2
    cv_n = 5
    gamma_horizon = 0.25
    mixing_values = [0.3, 0.01]

    # Analysis variables
    data_win_lens = [0.5]
    time_steps = [0.1]
    datasets = ["cvep"]
    algos = ["EEGnet_patchembeddingdilation"]
    baseline_methods = ["cvep_pvalue", "cvep_cumul"]

    exp = CVEPExperiment(
        epoch_len=epoch_len,
        o_cal_n=cv_n,
        test_n=test_n,
        data_win_lens=data_win_lens,
        time_steps=time_steps,
        datasets=datasets,
        algos=algos,
        baseline_methods=baseline_methods,
        init_delay=0.0,
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

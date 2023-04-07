"""
Utils for classification module

Author: Juan Jesús Torre, Ludovic Darmet, Giuseppe Ferraro
Mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr, Ludovic.DARMET@isae-supaero.fr, Giuseppe.FERRARO@isae-supaero.fr
"""

import numpy as np
import scipy.signal as scp

from scipy.signal import hilbert


def filterbank(data, sfreq, idx_fb, peaks):
    """
    Filter bank design for decomposing EEG data into sub-band components [1]

    Parameters
    ----------

    data: np.array, shape (trials, channels, samples) or (channels, samples)
        EEG data to be processed

    sfreq: int
        Sampling frequency of the data.

    idx_fb: int
        Index of filters in filter bank analysis

    peaks : list of len (n_classes)
        Frequencies corresponding to the SSVEP components.

    Returns
    -------

    y: np.array, shape (trials, channels, samples)
        Sub-band components decomposed by a filter bank

    Reference:
      [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using
           task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.

    Code based on the Matlab implementation from authors of [1]
    (https://github.com/mnakanishi/TRCA-SSVEP).
    """

    # Calibration data comes in batches of trials
    if data.ndim == 3:
        num_chans = data.shape[1]
        num_trials = data.shape[0]

    # Testdata come with only one trial at the time
    elif data.ndim == 2:
        num_chans = data.shape[0]
        num_trials = 1

    sfreq = sfreq / 2

    min_freq = np.min(peaks)
    max_freq = np.max(peaks)
    if max_freq < 40:
        top = 90
    else:
        top = 115
    diff = min_freq
    # Lowcut frequencies for the pass band (depends on the frequencies of SSVEP)
    # No more than 3dB loss in the passband
    passband = [min_freq + x * diff for x in range(7)]

    # At least 40db attenuation in the stopband
    if min_freq - 4 > 0:
        stopband = [
            min_freq - 4 + x * (diff - 2) if x < 3 else min_freq - 4 + x * diff
            for x in range(7)
        ]
    else:
        stopband = [2 + x * (diff - 2) if x < 3 else 2 + x * diff for x in range(7)]

    Wp = [passband[idx_fb] / sfreq, top / sfreq]
    Ws = [stopband[idx_fb] / sfreq, (top + 5) / sfreq]
    N, Wn = scp.cheb1ord(Wp, Ws, 3, 15)  # Chebyshev type I filter order selection.

    B, A = scp.cheby1(N, 0.5, Wn, btype="bandpass")  #  Chebyshev type I filter design

    y = np.zeros(data.shape)
    if num_trials == 1:  # For testdata
        for ch_i in range(num_chans):
            try:
                # The arguments 'axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1)' correspond
                # to Matlab filtfilt (https://dsp.stackexchange.com/a/47945)
                y[ch_i, :] = scp.filtfilt(
                    B,
                    A,
                    data[ch_i, :],
                    axis=0,
                    padtype="odd",
                    padlen=3 * (max(len(B), len(A)) - 1),
                )
            except Exception as e:
                print(e)
                print(f'Error on channel {ch_i}')
    else:
        for trial_i in range(num_trials):  # Filter each trial sequentially
            for ch_i in range(num_chans):  # Filter each channel sequentially
                y[trial_i, ch_i, :] = scp.filtfilt(
                    B,
                    A,
                    data[trial_i, ch_i, :],
                    axis=0,
                    padtype="odd",
                    padlen=3 * (max(len(B), len(A)) - 1),
                )
    return y


def _get_power_score(ress_comp, sfreq, target_freq, fwhm=0.5):
    """
    Get power from a RESS component

    Parameters
    ----------

    ress_comp : ndarray of shape (1, n_samples)
        Single RESS component.

    sfreq : float
        Sampling frequency of the data.

    target_freq : float
        Frequency at which the RESS component has been filtered, in Hz.

    fwhm : float, default=.5
        Full width at half-maximum for the gaussian filter.

    Returns
    -------

    power : float
        Power score of the target frequency.
    """

    filt_comp = gaussfilt(ress_comp, sfreq, f=target_freq, fwhm=fwhm)
    analytic_comp = hilbert(filt_comp, axis=0)
    power = np.mean(np.abs(analytic_comp) ** 2, axis=0)

    return power


def _get_cca_score(single_trial, sfreq, target_freq, cca, n_comp=1, n_harmonics=1):
    """
    Compute CCA correlation comparing the EEG signal to a template composed of ideal
    sine-cosine waves at the target frequency + n harmonics. Implementation follows
    procedure described on [1]
    """

    trial_len = single_trial.shape[0]
    t = np.arange(0, trial_len / sfreq, 1 / sfreq)

    # Find the harmonics you want for your target waves
    harmonics = list(range(n_harmonics, n_harmonics + 2))
    # Generate canonical sin and cos wave (and harmonics)
    all_waves = []
    for i in harmonics:
        freq = i * target_freq

        sin = np.sin(2 * np.pi * (freq) * t)
        cos = np.cos(2 * np.pi * (freq) * t)

        all_waves.append(sin)
        all_waves.append(cos)

    y = np.vstack(all_waves).T  # (samples, waves)

    # Fit CCA and get the cannonical correlation
    cca.fit(single_trial, y)
    x_scores_, y_scores_ = cca.transform(single_trial, y)

    corr_score = np.diag(
        np.corrcoef(x_scores_, y_scores_, rowvar=False)[:n_comp, n_comp:]
    )  # cca.x_scores_ is deprecated

    return corr_score


def schaefer_strimmer_cov(X):
    """Schaefer-Strimmer covariance estimator
    Shrinkage estimator using method from [1]:
    .. math::
            \hat{\Sigma} = (1 - \gamma)\Sigma_{scm} + \gamma T
    where :math:`T` is the diagonal target matrix:
    .. math::
            T_{i,j} = \{ \Sigma_{scm}^{ii} \text{if} i = j, 0 \text{otherwise} \}
    Note that the optimal :math:`\gamma` is estimate by the authors' method.
    :param X: Signal matrix, Nchannels X Nsamples
    :returns: Schaefer-Strimmer shrinkage covariance matrix, Nchannels X Nchannels
    References
    ----------
    [1] Schafer, J., and K. Strimmer. 2005. A shrinkage approach to
    large-scale covariance estimation and implications for functional
    genomics. Statist. Appl. Genet. Mol. Biol. 4:32.
    http://doi.org/10.2202/1544-6115.1175
    """
    _, Ns = X.shape[0], X.shape[1]
    C_scm = np.cov(X, ddof=0)
    X_c = X - np.tile(X.mean(axis=1), [Ns, 1]).T

    # Compute optimal gamma, the weigthing between SCM and srinkage estimator
    R = Ns / (Ns - 1.0) * np.corrcoef(X)
    var_R = (X_c ** 2).dot((X_c ** 2).T) - 2 * C_scm * X_c.dot(X_c.T) + Ns * C_scm ** 2
    var_R = Ns / ((Ns - 1) ** 3 * np.outer(X.var(axis=1), X.var(axis=1))) * var_R
    R -= np.diag(np.diag(R))
    var_R -= np.diag(np.diag(var_R))
    gamma = max(0, min(1, var_R.sum() / (R ** 2).sum()))

    return (1.0 - gamma) * (Ns / (Ns - 1.0)) * C_scm + gamma * (
        Ns / (Ns - 1.0)
    ) * np.diag(np.diag(C_scm))

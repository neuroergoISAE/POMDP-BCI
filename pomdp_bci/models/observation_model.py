"""
Defines the observation model of the POMDP agent.

The observation domain depends on the particular participant, as it is estimated in a data-driven way from the
calibration data used to train the classification algorithm that informs the POMDP. On its simplest form, the
confusion matrix of the classifier is used, as implemented in [1].

The observation space can also be discretized so the probabilities p(o|S) depend on the signal-to-noise ratio of
each individual, and can be obtained following the process detailed in [2].

The feature space is discretized using a gaussian mixture model for each class, then creating a high-resolution
grid and calculating p(o|S) for each point in the grid via numerical integration. Once that is done, observations
are grouped by making those that would result on the same belief update part of the same discrete class, taking
advantage of the fact that the values p(o|s) that compose each vector p(o|S) need to be normalized so their sum equals
one.

References:
    [1] - Singh, G., Roy, R.N., & Chanel, C.P.C. (2022).
          POMDP-based adaptive interaction through physiological computing.
          1st International Conference on Hybrid Human Artificial Intelligence.
    [2] - Bryan, M. J., Martin, S. A., Cheung, W., & Rao, R. P. (2013).
          Probabilistic co-adaptive brain–computer interfacing.
          Journal of neural engineering, 10(6), 066008.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py

from bci_pomdp.domain.observation import BCIObservation


class ObservationModel(pomdp_py.ObservationModel):
    """
    Class modeling the probability p(O | S', A). Since the model does not make any observations on terminal
    states and all transitions are deterministic, we can express it as p(O | S').

    Since the observations are given by each particular user's brain data, the number of observations can vary.
    Also, a discretization procedure is needed to fit the continuous feature space into a discrete observation
    space.

    Parameters
    ----------

    Features: n-D np.array
        Feature array for the observation matrix. Depending on the discretization method, the required features
        are different. 'Conf_matrix' requires a confusion matrix in form of an array of the shape (n_states, n_obs),
        'gaussian' and 'autoencoder' require a classic feature array of shape (observations, labels).

    Attributes
    ----------
    discretization: str, ['gaussian', 'autoencoder', 'conf_matrix']
        Method used to fit the continuous feature space. 'Gaussian' uses a gaussian mixture to find a continuous
        approximation of the observation space, and then carries on with the discretization procedure detailed
        above. 'Autoencoder' adds an extra dimensionality reduction step at the beginning using an autoencoder to
        extract the most meaningful information from the data, then performs the same discretization process.
        'Conf_matrix' takes the confusion matrix from a previously trained classifier and uses it as the observation
        matrix.

        For low-dimensionality feature extraction methods (with respect to the number of classes), that yield a
        single number per class, the gaussian discretization is suggested, as was done in the original paper.
        For other feature extraction methods with a higer dimensionality (e.g., using a whole EEG epoch as features),
        the autoencoder procedure is a good way to avoid excessive scaling of the observation space.

    observation_matrix: 2D np.array, (n_class, n_observations)
        Matrix representing the observation model, where each element represents the probability of obtaining
        the observation corresponding on the column given that the agent is currently at the state corresponding
        to the row. Example:
            observation_matrix[2][5] = p(O=o_5|S=s_2)
    """
    def __init__(self, features, discretization='conf_matrix'):
        self.discretization = discretization
        self.observation_matrix = self._make_obs_matrix(features)
        self.n_states, self.n_obs = self.observation_matrix.shape

    def _fit_autoencoder(self):
        """Apply an autoencoder to the features"""
        pass

    def _make_discretization(self, features):
        """Perform discretization of the observation space"""
        # Check if dim reduction is needed
        # if self.discretization == 'autoencoder':
        #     features = self._fit_autoencoder()

        # Fit gaussian

        # Make high-resolution grid

        # For every grid point o:
        #     1. Calculate p(o|S). Each p(o|S) should be an n_class-dimensional vector.
        #        They have to be normalized to sum 1. The output is an 'evidence_vector'
        #     2. Assign to each element of the vector an integer label in [1. n], where
        #        a higher n corresponds to a higher discretization resolution. To label
        #        the ith element of the vector (evidence_i):
        #            label_i = round((n - 1), * evidence_i + 1)
        #
        # This outputs a label vector for each point of the grid. Finally, a discrete
        # observation label is assigned to every unique value these vectors take

        # k-means clustering among the discretized areas

        # Compute the observation probabilities p(o|S) by performing numerical integration within
        # each of the regions for each class' Gaussian probability density Function (PDF)

        # Make the observation matrix
        raise NotImplementedError

    def _make_obs_matrix(self, features):
        """
        Fit the continuous feature space into a discrete observation space.

        Parameters
        ----------
        features: n-D np.array (n_observations, ...) --> PLACEHOLDER (need to see how the features work in practice)
            Features from the participant. The number of observations should be the first dimension, and the next
            dimension (in case of correlation vectors) or dimensions (in case of complete EEG epochs) should be
            the features
        """

        if self.discretization == 'conf_matrix':
            observation_matrix = features
        elif self.discretization == 'gaussian':
            observation_matrix = self._make_discretization(features)
        elif self.discretization == 'autoencoder':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid discretization: {self.discretization}. Action must be an integer or 'wait'")

        return observation_matrix

    def get_observation_label(self, features):
        """
        Get the label corresponding to a new observation

        Parameters
        ----------

        features: np.array
            New observation

        Returns
        -------

        observation_label: int
            Label corresponding to the observation in the discretized space
        """
        pass

    def probability(self, observation, next_state, action):
        # The probability of obtaining a new observation knowing the state is given by the discretization / conf matrix
        obs_idx = int(observation.id)
        state_idx = int(next_state.id)
        return self.observation_matrix[state_idx][obs_idx]

    def get_all_observations(self):
        return [BCIObservation(o) for o in range(self.n_obs)]





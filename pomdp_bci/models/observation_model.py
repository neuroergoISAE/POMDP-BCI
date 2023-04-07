"""
Defines the observation model of the POMDP agent.

The observation domain depends on the particular participant, as it is estimated in a data-driven way from the
calibration data used to train the classification algorithm that informs the POMDP. On its simplest form, the
confusion matrix of the classifier is used, as implemented in [1].

References:
    [1] - Singh, G., Roy, R.N., & Chanel, C.P.C. (2022).
          POMDP-based adaptive interaction through physiological computing.
          1st International Conference on Hybrid Human Artificial Intelligence.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py

from bci_pomdp.domain.observation import BCIObservation


class ObservationModel(pomdp_py.ObservationModel):
    """
    Class modeling the probability p(O | S', A). Since the model does not make any observations on terminal
    states and all transitions are deterministic, we can express it as p(O | S').

    Parameters
    ----------
    conf_matrix: n-D np.array
        Requires a confusion matrix in form of an array of the shape (n_states, n_obs) from a previously trained
        classifier and uses it as the observation matrix.

    Attributes
    ----------
    observation_matrix: 2D np.array, (n_class, n_observations)
        Matrix representing the observation model, where each element represents the probability of obtaining
        the observation corresponding on the column given that the agent is currently at the state corresponding
        to the row. Example:
            observation_matrix[2][5] = p(O=o_5|S=s_2)
    """
    def __init__(self, conf_matrix):
        self.observation_matrix = conf_matrix
        self.n_states, self.n_obs = self.observation_matrix.shape

    def probability(self, observation, next_state, action):
        # The probability of obtaining a new observation knowing the state is given by the discretization / conf matrix
        obs_idx = int(observation.id)
        state_idx = int(next_state.id)
        return self.observation_matrix[state_idx][obs_idx]

    def get_all_observations(self):
        return [BCIObservation(o) for o in range(self.n_obs)]





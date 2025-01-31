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
import numpy as np

from pomdp_bci.domain.observation import BCIObservation


class ObservationModel(pomdp_py.ObservationModel):
    """
    Class modeling the probability p(O | S', A). Since the model does not make any observations on terminal
    states and all transitions are deterministic, we can express it as p(O | S').

    Parameters
    ----------
    conf_matrix: 2D np.array
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
        if "wait" in action.name:
            # The probability of obtaining a new observation knowing the state is given by the discretization / conf matrix
            obs_idx = int(observation.id)
            state_idx = int(next_state.id)
            return self.observation_matrix[state_idx][obs_idx]
        else:  # When a decision is taken, the observation is provided at random
            return 1 / self.n_states

    def sample(self, next_state, action):
        if "wait" in action.name:
            state_idx = next_state.id
            obs_p = self.observation_matrix[state_idx]
        else:
            obs_p = None

        return np.random.choice(self.get_all_observations(), p=obs_p)

    def get_all_observations(self):
        return [BCIObservation(o) for o in range(self.n_obs)]


class FiniteObservationModel(ObservationModel):
    """
    Similar to ObservationModel, with the inclusion of a terminal state (finite-horizon problem), to which
    the model transitions to when an action is taken. This requires a terminal observation to be included,
    which is deterministically observed when in or when transitioning to the terminal state.

    Parameters
    ----------
    conf_matrix: 2D np.array
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
        super().__init__(conf_matrix)

    def probability(self, observation, next_state, action):
        if "term" in observation.name:  # Terminal observation
            if (
                "term" in next_state.name or "wait" not in action.name
            ):  # Transition to terminal state
                return 1
            else:
                return 0
        else:  # Non-terminal observation
            if "term" in next_state.name or "wait" not in action.name:
                return 0
            else:
                return super().probability(observation, next_state, action)

    def sample(self, next_state, action):
        if (
            "term" in next_state.name or "wait" not in action.name
        ):  # Transition to terminal state
            return BCIObservation("term")
        else:  # Other transitions
            super().sample(next_state, action)

    def get_all_observations(self, include_terminal=True):
        all_obs = [BCIObservation(o) for o in range(self.n_obs)]
        if include_terminal:
            all_obs.append(BCIObservation("term"))

        return all_obs


class TDObservationModel(ObservationModel):
    """
    Time-dependent extension of the ObservationModel class that takes into account
    the time-step within each POMDP trial and allows the observation function to have
    different observation probabilities depending on the time step.

    This allows the time-dependent POMDP to leverage the fact that the more a trial
    advances, the more brain data from the subject is available. Thus, the probability
    p(o | s, a, d) (where d is the time step) should be less uncertain the longer the
    trial is.

    This also removes the contraint present in the basic model where the
    initial time step of each trial needs a sufficiently large brain data window to
    yield good classification (e.g. 0.5s), since restrictions on the previous observation
    function required all time steps to use data windows of the same length.

    Lastly, this extension includes a terminal observation o_term, which is deterministically
    observed at the terminal state.

    Parameters
    ----------

    conf_matrix: 3D np.array, shape (n_steps, n_states, n_observations)
        Array containing the confusion matrices for each time-step of the model. Each 2D matrix must
        be of shape (n_true, n_pred), corresponding to (n_states, n_observations).

    Attributes
    ----------

    observation_matrix: 3D np.array, (n_timesteps, n_class, n_observation)
        Matrix representing the observation model, where each element represents the probability of obtaining
        the observation corresponding on the third dimension given that the agent is currently at the state
        corresponding to the second simension and the current time step of the trial is that of the first dimension.

        Example:
            observation_matrix[3][2][5] = p(o=o_5|s=s_2, d=3)
    """

    def __init__(self, conf_matrix):
        self.observation_matrix = conf_matrix
        self.n_steps, self.n_states, self.n_obs = self.observation_matrix.shape

    def probability(self, observation, next_state, action):
        if "term" in observation.name:  # Terminal observation
            if (
                "term" in next_state.name or "wait" not in action.name
            ):  # Transition to terminal state
                return 1
            else:
                return 0
        else:  # Non-terminal observation
            if "term" in next_state.name or "wait" not in action.name:
                return 0
            else:
                obs_idx = observation.id
                state_idx = next_state.id
                state_step = (
                    next_state.t - 1
                )  # observation_matrix[0] corresponds to when next_state.t is 1
                return self.observation_matrix[state_step][state_idx][obs_idx]

    def sample(self, next_state, action):
        if (
            "term" in next_state.name or "wait" not in action.name
        ):  # Transition to terminal state
            return BCIObservation("term")
        else:  # Other transitions
            state_idx = next_state.id
            state_step = (
                next_state.t - 1
            )  # observation_matrix[0] corresponds to when next_state.t is 1
            obs_p = self.observation_matrix[state_step][state_idx]
            return np.random.choice(
                self.get_all_observations(include_terminal=False), p=obs_p
            )

    def get_all_observations(self, include_terminal=True):
        all_obs = [BCIObservation(o) for o in range(self.n_obs)]
        if include_terminal:
            all_obs.append(BCIObservation("term"))

        return all_obs

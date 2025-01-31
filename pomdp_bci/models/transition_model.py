"""
Define the transition model for an n-flicker SSVEP problem.

We assume the user keeps looking at the same flicker until an action is taken. After that, we
assume an equal probability of the participant looking at the next flicker.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py
import random
import itertools

import numpy as np

from pomdp_bci.domain import BCIState, TDState


class TransitionModel(pomdp_py.TransitionModel):
    """
    Transition model for the SSVEP problem.

    This model assumes that the state does not change until the agent executes
    a decision action (i.e. any action other than 'wait'):

        p(s'|s, a_wait) = 1  for s'_id == s_id
                          0  for s'_id != s_id

    When a decision is taken, the state can transition to any other state
    with uniform probability:

        p(s'|s, a) = 1 / N  if a != a_wait,

    where N is the number of states / targets.


    Attributes
    ----------

    n_targets: int
        Total number of states. Used for generating state list and
        giving uniform probability of state change when an action
        is taken
    """

    def __init__(self, n_targets):
        if not isinstance(n_targets, int):
            raise TypeError(
                f"Invalid number of states: {n_targets}. It must be an integer."
            )
        self.n_targets = n_targets

    def probability(self, next_state, state, action):
        """Returns the probability p(s'|s, a)"""
        if "wait" in action.name:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9 / (self.n_targets - 1)
        else:
            return 1 / self.n_targets

    def sample(self, state, action):
        """Randomly samples next state according to transition model"""
        if "wait" in action.name:
            return state
        else:
            return random.choice(self.get_all_states())

    def get_all_states(self):
        """Returns a list of all states"""
        return [BCIState(s) for s in range(self.n_targets)]


class TDTransitionModel(TransitionModel):
    """
    Transition model extension for time-dependent POMDP. Takes into account
    the state's time step in order to introduce the following differences:

      - The action wait transitions deterministically to the same state
        but with a time-step increase of 1 so that
          p(s'|s, a_wait) = 1 for s_id == s'_id and s_t = d, s'_t = d+1
          p(s'|s, a_wait) = 0 otherwise

      - In the case where any action other than wait is taken, or when
        the time step is the maximum value (known beforehand), the model
        transitions to the terminal state with probability 1:

          p(s'|s, a_wait) = 1 for s' = s_term, s_t = D
          p(s'|s, a_wait) = 0 for s' != s_term, s_t = D

          p(s'|s, a) = 1 for all a_id != a_wait, s' = s_term, all s_t
          p(s'|s, a) = 1 for all a_id != a_wait, s' != s_term, all s_t

        With N being the total number of targets and D the total number of time
        steps, with 'id' being the number associated with a target (id = 1, ..., N)
        and 't' being the time step associated with the state (t = 1, ... D).
        The total number of states for the time-dependent problem is N * D.

    Attributes
    ----------

    n_targets: int
        Total number of targets / actionable commands. In a time-dependent problem, the
        number of states is not equivalent to the number of BCI targets. Used for
        generating state list and giving uniform probability of state change when an action
        is taken

    n_steps: int
        Total number of time steps. Used to limit the states to which
        actions other than wait can transition
    """

    def __init__(self, n_targets, n_steps):
        super().__init__(n_targets)

        if not isinstance(n_steps, int):
            raise TypeError(
                f"Invalid number of steps: {n_steps}. It must be an integer."
            )
        self.n_steps = n_steps
        self.max_t = self.n_steps - 1  # To handle 0 indexing of states and time steps

    def probability(self, next_state, state, action):
        """Returns the probability p(s'|s, a)"""
        # If the current state is the terminal state, transition to itself with probability 1
        if "term" in state.name:
            if "term" in next_state.name:
                return 1.0
            else:
                return 0.0
        else:  # Not terminal state
            if "wait" not in action.name:  # Decision -> terminal state
                if "term" in next_state.name:
                    return 1
                else:
                    return 0
            elif state.t == self.max_t:  # Wait at last time step -> terminal_state
                if "term" in next_state.name:
                    return 1
                else:
                    return 0
            else:  # Wait at time steps other than the last one -> Next time step (almost surely)
                if next_state.t == state.t + 1:  # For the next time step
                    if next_state.id == state.id:
                        return 1.0 - 1e-9
                    else:  # Other states in the next time step
                        return 1e-9 / (self.n_targets - 1)
                else:  # Can't travel through time... yet
                    return 0

    def sample(self, state, action):
        """Randomly samples next state according to transition model"""
        # Always sample the terminal state if current state is terminal
        if "term" in state.name:
            return TDState("term", 0)
        else:  # Not terminal state
            if "wait" not in action.name:  # Decision -> terminal state
                return TDState("term", 0)
            elif (
                state.t == self.max_t
            ):  # Wait action on last time step -> Same state at t = max
                return TDState("term", 0)
            else:  # Wait action on time steps other than the last one -> Next time step (almost surely)
                next_step = state.t + 1
                possible_states = self.get_all_states(t_step=next_step)
                proba = [self.probability(s, state, action) for s in possible_states]
                return np.random.choice(possible_states, p=proba)

    def get_all_states(self, t_step=None):
        """Returns a list of all states"""
        if t_step is not None:  # Get list of states for a given time_step
            all_states = [TDState(s, t_step) for s in range(self.n_targets)]
        else:  # All states, all time steps (including the terminal state)
            all_states = [
                TDState(s, d)
                for s, d in itertools.product(
                    range(self.n_targets), range(self.n_steps)
                )
            ]
            all_states.append(TDState("term", 0))
        return all_states

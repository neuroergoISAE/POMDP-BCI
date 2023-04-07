"""
Define the transition model for an n-flicker SSVEP problem.

We assume the user keeps looking at the same flicker until an action is taken. After that, we
assume an equal probability of the participant looking at the next flicker.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py
import random

from pomdp_bci.domain import BCIState


class TransitionModel(pomdp_py.TransitionModel):
    """
    Transition model for the SSVEP problem.

    Attributes
    ----------

    n_states: int
        Total number of states. Used for generating state list and
        giving uniform probability of state change when an action
        is taken
    """
    def __init__(self, n_states):
        if not isinstance(n_states, int):
            raise TypeError(f"Invalid number of states: {n_states}. It must be an integer.")
        self.n_states = n_states

    def probability(self, next_state, state, action):
        """Returns the probability p(s'|s, a)"""
        if "wait" in action.name:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9
        else:
            return 1 / self.n_states

    def sample(self, state, action):
        """Randomly samples next state according to transition model"""
        if "wait" in action.name:
            return state
        else:
            return random.choice(self.get_all_states())

    def get_all_states(self):
        """Returns a list of all states"""
        return [BCIState(s) for s in range(self.n_states)]

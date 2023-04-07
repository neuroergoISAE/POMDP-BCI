"""
Define the policy model for an n-flicker SSVEP problem.

We assume no prior knowledge of action preference, as flickers can be embedded into any kind of interface.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py
import random

from pomdp_bci.domain import BCIAction


class PolicyModel(pomdp_py.RolloutPolicy):
    """
    Simple policy model with uniform prior

    Attributes
    ----------

    n_states: int
        Total number of states. Used to list every action
        as there is one action per flicker
    """
    def __init__(self, n_states):
        if not isinstance(n_states, int):
            raise TypeError(f"Invalid number of states: {n_states}. It must be an integer.")
        self.n_states = n_states

    def sample(self, state):
        """Return an action with uniform probability"""
        return random.choice(self.get_all_actions())

    def rollout(self, state, *args):
        """Needed for interface compliance. Uses self.sample()"""
        return self.sample(state)

    def get_all_actions(self):
        """Return a list of all possible actions"""
        all_actions = [BCIAction(a) for a in range(self.n_states)]
        all_actions.append(BCIAction("wait"))
        return all_actions

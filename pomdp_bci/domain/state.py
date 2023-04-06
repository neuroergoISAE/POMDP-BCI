"""
Defines the state of the POMDP agent, based on the number of flickers.

Description: A human participant is looking at one of the N available flickers (states). The goal of the agent
is to determine which flicker the participant is looking at and execute an action that moves the model to a terminal
state associated with that action.

State domain: {F1, F2, ..., Fn}, where n equals the number of flickers.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py


class BCIState(pomdp_py.State):
    def __init__(self, state_n):
        """
        Parameters
        ----------
        state_n: int
            The flicker number corresponding to the state. This class does not
            check if this number is within bounds of the state space or not.
        """
        if not isinstance(state_n, int):
            raise TypeError(f"Invalid state index: {state_n}. States are indexed by integers")

        self.id = state_n
        self.name = f"s_{state_n}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, BCIState):
            return self.name == other.name
        else:
            return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"BCIState({self.name})"



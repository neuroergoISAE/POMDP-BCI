"""
Defines the observations of the POMDP agent, which are based on the flickers that are presented.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py


class BCIObservation(pomdp_py.Observation):
    def __init__(self, obs_n):
        """
        Parameters
        ----------
        obs_n: int
            Observation number. These are identified by a single number, since the total number of observations
            and their probabilities can vary from case to case. Note: This object does not check whether the
            observation is within the boundaries of all possible observations
        """
        if not isinstance(obs_n, int):
            raise TypeError(f"Invalid observation index: {obs_n}. Observations are indexed by integers")

        self.id = obs_n
        self.name = f"o_{obs_n}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, BCIObservation):
            return self.name == other.name
        else:
            return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"BCIObservation({self.name})"

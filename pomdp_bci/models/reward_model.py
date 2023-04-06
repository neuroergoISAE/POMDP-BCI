"""
Define the reward model for an n-flicker SSVEP problem.

In its simplest form, the reward model assigns a cost to mistakes, a reward to correct actions and a small cost
to the 'wait' action. The interface is made in a way that allows the user to modify the values for these three
parameters in order to facilitate reward engineering when learning the model.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py


class RewardModel(pomdp_py.RewardModel):
    """
    Reward model for the SSVEP problem. Note: The code does not check whether attributes are positive or negative
    numbers, nor whether it is bigger or smaller than the other values.

    Attributes
    ----------

    hit_reward: int, default 10
        Amount to reward the model when it takes correct actions.

    miss_cost: int, default -100
        Amount to penalize the model when it makes mistakes

    wait_cost: int, default -1
        Amount to penalize the model for the 'wait'/passive action
    """
    def __init__(self, hit_reward=10, miss_cost=-100, wait_cost=-1):
        if not all(isinstance(attr, int) for attr in [hit_reward, miss_cost, wait_cost]):
            raise TypeError("All cost/reward values must be integers.")

        self.hit_reward = hit_reward
        self.miss_cost = miss_cost
        self.wait_cost = wait_cost

    def _reward_func(self, state, action):
        """
        The correct action is assumed to be the one that shares ID (i.e., number) with a given state,
        since we assume that each flicker is embedded in a button or actuator.
        """
        if 'wait' in action.name:
            return self.wait_cost
        else:
            if action.id == state.id:  # HIT
                return self.hit_reward
            else:  # MISS
                return self.miss_cost

    def sample(self, state, action, next_state):
        """Deterministic reward"""
        return self._reward_func(state, action)

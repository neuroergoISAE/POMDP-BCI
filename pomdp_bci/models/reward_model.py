"""
Define the reward model for an n-flicker SSVEP problem.

In its simplest form, the reward model assigns a cost to mistakes, a reward to correct actions and a small cost
to the 'wait' action. The interface is made in a way that allows the user to modify the values for these three
parameters in order to facilitate reward engineering when learning the model.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py

import numpy as np


class RewardModel(pomdp_py.RewardModel):
    """
    Reward model for the SSVEP problem.

    The model estipulates a cost (negative reward) for the action 'wait'. The
    reward for any other actions is determined by the state:

        r(s, a) = hit_reward  if s_id == a_id
                = miss_cost   otherwise

    Note: The code does not check whether attributes are positive or negative
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
        if not all(
            isinstance(attr, int) for attr in [hit_reward, miss_cost, wait_cost]
        ):
            raise TypeError("All cost/reward values must be integers.")

        self.hit_reward = hit_reward
        self.miss_cost = miss_cost
        self.wait_cost = wait_cost

    def _reward_func(self, state, action):
        """
        The correct action is assumed to be the one that shares ID (i.e., number) with a given state,
        since we assume that each flicker is embedded in a button or actuator.
        """
        if "wait" in action.name:
            return self.wait_cost
        else:
            if action.id == state.id:  # HIT
                return self.hit_reward
            else:  # MISS
                return self.miss_cost

    def sample(self, state, action, next_state):
        """Deterministic reward"""
        return self._reward_func(state, action)


class TermRewardModel(RewardModel):
    """
    Extension of the reward model that includes a reward of 0 for an absorbent terminal
    state s_term. This is the reward model used for the time-dependent problem.

    Attributes
    ----------

    hit_reward: int, default 10
        Amount to reward the model when it takes correct actions.

    miss_cost: int, default -100
        Amount to penalize the model when it makes mistakes.

    wait_cost: int, default -1
        Amount to penalize the model for the 'wait'/passive action.

    constant_cost: bool, default False
        If true, trials where the model makes a mistake have the same cost, irrespective from
        the duration of the trial. This is achieved by adding the combined wait cost of the
        trial at the time of making a mistake.
    """

    def __init__(self, hit_reward=10, miss_cost=-100, wait_cost=-1):
        if not all(
            isinstance(attr, int) for attr in [hit_reward, miss_cost, wait_cost]
        ):
            raise TypeError("All cost/reward values must be integers.")

        super().__init__(
            hit_reward=hit_reward, miss_cost=miss_cost, wait_cost=wait_cost
        )

    def _reward_func(self, state, action):
        """
        The correct action is assumed to be the one that shares ID (i.e., number) with a given state,
        since we assume that each flicker is embedded in a button or actuator. Any action on the
        terminal state gives a reward of 0.
        """
        if "term" in state.name:
            return 0
        else:
            return super()._reward_func(state, action)

    def sample(self, state, action, next_state):
        """Deterministic reward"""
        return self._reward_func(state, action)

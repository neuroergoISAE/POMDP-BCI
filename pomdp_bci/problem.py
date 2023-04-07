"""
BCI problem definition using a POMDP model.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py

from bci_pomdp.models import ObservationModel, TransitionModel, PolicyModel, RewardModel


class BCIProblem(pomdp_py.POMDP):
    """
    Class that defines the VEP problem. Used to pass required arguments to the models
    before initializing the POMDP class with agent and environment.

    Parameters
    ----------

    init_belief: pomdp_py.Histogram
        Initial probability distribution over the belief state, used for the Agent object

    init_true_state: BCIState object
        Initial state of the problem, used for the Environment object

    n_class: int
        Number of flickers of this problem. Used for PolicyModel and TransitionModel

    conf_matrix: n-D np.array
        Confusion matrix obtained from a previously trained classifier to use as the observation matrix.
        See bci_pomdp.ObservationModel for more information

    hit_reward: int
        Reward for correct actions. See bci_pomdp.TransitionModel for more information

    miss_cost: int
        Cost for incorrect actions. See bci_pomdp.TransitionModel for more information

    wait_cost: int
        Cost for the 'wait' action. See bci_pomdp.TransitionModel for more information
    """
    def __init__(self, init_belief, init_true_state, n_class, conf_matrix,
                 hit_reward=10, miss_cost=-100, wait_cost=-1):
        policy_model = PolicyModel(n_class)
        transition_model = TransitionModel(n_class)
        observation_model = ObservationModel(conf_matrix=conf_matrix)
        reward_model = RewardModel(hit_reward, miss_cost, wait_cost)

        agent = pomdp_py.Agent(init_belief,
                               policy_model,
                               transition_model,
                               observation_model,
                               reward_model)

        env = pomdp_py.Environment(init_true_state,
                                   transition_model,
                                   reward_model)

        super().__init__(agent, env, name=BCIProblem)

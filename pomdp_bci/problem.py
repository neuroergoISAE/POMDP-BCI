"""
VEP problem definition using a POMDP model.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py

from bci_pomdp.models import (
    TransitionModel,
    TDTransitionModel,
    ObservationModel,
    FiniteObservationModel,
    TDObservationModel,
    RewardModel,
    TermRewardModel,
    PolicyModel,
)


class BaseProblem(pomdp_py.POMDP):
    """
    Class that defines the VEP problem. Used to pass required arguments to the models
    before initializing the POMDP class with agent and environment.

    Parameters
    ----------

    init_belief: pomdp_py.Histogram
        Initial probability distribution over the belief state, used for the Agent object

    init_true_state: BCIState object
        Initial state of the problem, used for the Environment object

    n_targets: int
        Number of flickers of this problem. Used for PolicyModel and TransitionModel

    conf_matrix: 2D np.array
        Confusion matrix obtained from a previously trained classifier to use as the observation matrix.
        See bci_pomdp.ObservationModel for more information

    hit_reward: int
        Reward for correct actions. See bci_pomdp.TransitionModel for more information

    miss_cost: int
        Cost for incorrect actions. See bci_pomdp.TransitionModel for more information

    wait_cost: int
        Cost for the 'wait' action. See bci_pomdp.TransitionModel for more information
    """

    def __init__(
        self,
        init_belief,
        init_true_state,
        n_targets,
        conf_matrix,
        hit_reward=10,
        miss_cost=-100,
        wait_cost=-1,
    ):
        policy_model = PolicyModel(n_targets)
        transition_model = TransitionModel(n_targets)
        observation_model = ObservationModel(conf_matrix=conf_matrix)
        reward_model = RewardModel(hit_reward, miss_cost, wait_cost)
        name = "BCIProblem"

        agent = pomdp_py.Agent(
            init_belief, policy_model, transition_model, observation_model, reward_model
        )

        env = pomdp_py.Environment(init_true_state, transition_model, reward_model)

        super().__init__(agent, env, name=name)


class TDProblem(pomdp_py.POMDP):
    """
    Time-dependent BCI problem. It constructs the problem with time-dependent transition
    and observation models, as well as a reward model that includes terminal state (also
    included in time-dependent models).

    Parameters
    ----------

    conf_matrix: 2D or 3D np.array
        Confusion matrix for the time-dependent observation model. Needs to be 3D with shape
        (n_steps, n_states, n_observations) for td_obs=True and 2D with shape (n_states,
        n_observations) for td_obs=False.
            See bci_pomdp.TDObservationModel and bci_pomdp.ObservationModel for more information.

    n_steps: int or None, default=None
        Number of time steps for a time-dependent BCI-POMDP problem. It usually entails the
        number of decision steps made in one trial plus one step where the POMDP makes a final
        action after all data has been used.

    td_obs: bool, default=True
        Variable to control the type of observation model used. If true, the observation model
        will depend on time (TDObservationModel), whereas if it is false, it will not
        (ObservationModel). Note that the different base and time-dependent models expect confusion
        matrices of 3 and 2 dimensions, respectively.
    """

    def __init__(
        self,
        init_belief,
        init_true_state,
        n_targets,
        conf_matrix,
        n_steps,
        td_obs=True,
        hit_reward=10,
        miss_cost=-100,
        wait_cost=-1,
    ):

        reward_model = TermRewardModel(hit_reward, miss_cost, wait_cost)
        policy_model = PolicyModel(
            n_targets
        )  # Policy model makes the action list using the number of targets
        transition_model = TDTransitionModel(n_targets, n_steps)
        if td_obs:
            observation_model = TDObservationModel(conf_matrix=conf_matrix)
            name = "Time-dependent BCIProblem"
        else:
            observation_model = FiniteObservationModel(conf_matrix=conf_matrix)
            name = "Time-mixed BCIProblem"

        agent = pomdp_py.Agent(
            init_belief, policy_model, transition_model, observation_model, reward_model
        )

        env = pomdp_py.Environment(init_true_state, transition_model, reward_model)

        super().__init__(agent, env, name=name)

"""
Defines the actions of the POMDP agent, based on the number of flickers.

Action domain: {a_1, a_2, ..., a_n, a_wait}, where n equals the number of targets. From all possible initial states,
all actions are possible, a_x takes the agent to from any state F_y to the state K_x, whereas a_wait stays in the same
state.

Author: Juan Jesus Torre Tresols
mail: Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import pomdp_py


class BCIAction(pomdp_py.Action):
    def __init__(self, action_name):
        """
        Parameters
        ----------
        action_name: int or str ['wait']
            Actions are organized in numbers, an action with a given number takes the agent from any state
            to the terminal corresponding to said number. The action 'wait' always transitions to the same 
            state the agent was previously in. Note: This class does not check whether the action number is
            within bounds of the action space.
        """
        if not isinstance(action_name, int):
            if action_name == 'wait':
                pass
            else:
                raise TypeError(f"Invalid action: {action_name}. Action must be an integer or 'wait'")

        self.id = action_name
        self.name = f"a_{action_name}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, BCIAction):
            return self.name == other.name
        else:
            return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"BCIAction({self.name})"

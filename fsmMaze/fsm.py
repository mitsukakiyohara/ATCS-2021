"""
This module implements a Finite State Machine (FSM).

You define an FSM by building tables of transitions. For a given input symbol
the process() method uses these tables to decide what action to call and what
the next state will be. The FSM has a table of transitions that associate the tuples:

        (input_symbol, current_state) --> (action, next_state)

Where "action" is a function you define. The symbols and states can be any
objects. You use the add_transition() method to add to the transition table.

@author: Ms. Namasivayam (TODO: replace with your name)
@version: 2022
"""


class FSM:
    def __init__(self, initial_state):
        # TODO: Initialize state transitions and current state
        # Dictionary (input_symbol, current_state) --> (action, next_state).

    def add_transition(self, input_symbol, state, action=None, next_state=None):
        """
        TODO: implement add transition
        This adds a transition that associates:
                (input_symbol, current_state) --> (action, next_state)

        The action may be set to None in which case the process() method will
        ignore the action and only set the next_state.

        The next_state may be set to None in which case the current state will be unchanged.
        """

    def get_transition(self, input_symbol, state):
        """
        TODO: Implement get transition
        This returns tuple (action, next state) given an input_symbol and state.
        Normally you do not call this method directly. It is called by
        process().
        """

    def process(self, input_symbol):
        """
        TODO: Implement process
        This is the main method that you call to process input. This may
        cause the FSM to change state and call an action. This method calls
        get_transition() to find the action and next_state associated with the
        input_symbol and current_state. If the action is None then the action
        is not called and only the current state is changed. This method
        processes one complete input symbol.
        """

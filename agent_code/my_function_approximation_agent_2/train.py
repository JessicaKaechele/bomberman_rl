import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
ACTIONS = {'UP': 0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.old_feature_vals = np.zeros((8))
    self.all_rewards = []



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    reward = reward_from_events(self, events)
    self.all_rewards.append(reward)

    if old_game_state:
        features = state_to_features(old_game_state)

        next_features = state_to_features(new_game_state)
        q_values_next = np.array([m.predict([next_features])[0] for m in self.model])
        update = reward + DISCOUNT_FACTOR * np.max(q_values_next)
        self.model[ACTIONS[self_action]].partial_fit([features], [update])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the model
    with open("jessi-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    with open("rewards/statistics", "a") as f:
        for reward in self.all_rewards:
            f.writelines(str(reward)+ "\n")
    self.all_rewards = []


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -10,
        e.INVALID_ACTION: -10,
        e.BOMB_EXPLODED: 0,
        e.BOMB_DROPPED: 1,
        e.CRATE_DESTROYED: 50,
        e.COIN_FOUND: 10,
        e.COIN_COLLECTED: 1000,
        e.KILLED_OPPONENT: 1000,
        e.KILLED_SELF: -1200,
        e.GOT_KILLED: -1000,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

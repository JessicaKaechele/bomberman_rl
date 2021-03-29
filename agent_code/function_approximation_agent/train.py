from collections import deque
from typing import List
import numpy as np

import events as e
from .features import state_to_features

from .custom_events import not_moving_event, did_not_escape_event, can_not_escape_event, bomb_near_crate_event, \
    coin_reachable_event, DISTANCE_2, DISTANCE_1, DISTANCE_0, DROP_BOMB_NEAR_CRATE, DID_NOT_ESCAPE
from .utils import add_statistics, end_statistics, save_model, save_rewards

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
ACTIONS = {'UP': 0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # variables for custom events
    self.dropped_bomb = False
    self.last_positions = deque(maxlen=5)

    # statistics
    self.collected_coins_episode = 0
    self.destroyed_crates_episode = 0
    self.dropped_bombs = 0
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
    x,y = new_game_state['self'][3]
    coins = new_game_state['coins']
    arena = new_game_state['field']

    add_statistics(self, events)

    add_events(arena, coins, events, new_game_state, self, x, y)

    reward = get_reward(self, events, new_game_state)
    self.all_rewards.append(reward)

    do_learning(self, new_game_state, old_game_state, reward, self_action)


def add_events(arena, coins, events, new_game_state, self, x, y):
    not_moving_event(x, y, events, self)
    coin_reachable_event(arena, coins, events, x, y)
    bomb_near_crate_event(arena, events, x, y)
    can_not_escape_event(events, new_game_state, x, y)
    did_not_escape_event(self, x, y, events)


def do_learning(self, new_game_state, old_game_state, reward, self_action):
    if old_game_state and new_game_state:
        features = state_to_features(old_game_state)
        next_features = state_to_features(new_game_state)
        q_values_next = np.array([m.predict([next_features])[0] for m in self.model])

        # bellman
        update = reward + DISCOUNT_FACTOR * np.max(q_values_next)
        self.model[ACTIONS[self_action]].partial_fit([features], [update])

        # q learning with temporal difference
        #q_value = self.model[ACTIONS[self_action]].predict([features])
        #new_q_value = np.max(q_values_next)
        #update = q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * new_q_value - q_value)
        #self.model[ACTIONS[self_action]].partial_fit([features], update)

    elif not new_game_state:
        features = state_to_features(old_game_state)
        self.model[ACTIONS[self_action]].partial_fit([features], [reward])



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

    reward = get_reward(self, events)
    self.all_rewards.append(reward)
    do_learning(self, None, last_game_state, reward, last_action)

    # reset this value if bomb dropped before dead
    self.dropped_bomb = False

    save_model(self)
    end_statistics(self, events)
    save_rewards(self)

def distances_to_bomb(bombs, position):
    x, y = position
    distances = 0
    for bomb in bombs:
        bomb_x =  bomb[0][0]
        bomb_y = bomb[0][1]
        if (x != bomb_x and y != bomb_y):
            distance = 15
        else:
            distance = np.sqrt((x - bomb_x) ** 2 + (y - bomb_y) ** 2)
        distances += distance
    return distances * 50

def get_reward(self, events, game_state = None):
    reward = reward_from_events(self, events)
    if game_state:
        reward += distances_to_bomb(game_state["bombs"], game_state['self'][3])
    return reward

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.BOMB_EXPLODED: 0,
        e.BOMB_DROPPED: 40,
        e.CRATE_DESTROYED: 30,
        e.COIN_FOUND: 10,
        e.COIN_COLLECTED: 1000,
        e.KILLED_OPPONENT: 1000,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0,
        # DROP_BOMB_NEAR_CRATE: 15,
        # DID_NOT_ESCAPE: -50,
        # CAN_NOT_ESCAPE:-100
        # COIN_NOT_REACHABLE:-20,
        # DISTANCE_2:-1,
        # DISTANCE_1:-2,
        # DISTANCE_0:-3
    }
    reward_sum = 0
    for event in events:
        if event == DID_NOT_ESCAPE:
            pass
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

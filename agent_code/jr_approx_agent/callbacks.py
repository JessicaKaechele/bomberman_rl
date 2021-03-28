from collections import deque
from datetime import datetime
import logging
import sys
import os

import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

POSSIBLE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'weights.pt'

EXPLORATION_RATE = 0.75
MIN_EXPLORATION_RATE = 0.15


FEATURE_LENGTH = 496
# FEATURE_LENGTH = 30*3


def setup(self):
    self.is_fit = False
    self.last_act_was_exploration = False

    if self.train:
        self.exploration_rate = EXPLORATION_RATE
    else:
        self.exploration_rate = 0.05

    if not self.train:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    file_handler = logging.FileHandler(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    file_handler.setLevel(logging.DEBUG)
    self.logger.addHandler(file_handler)

    if not self.train and os.path.exists(MODEL_FILE):
        self.logger.info("Using existing model to play")
        with open(MODEL_FILE, "rb") as file:
            self.q = pickle.load(file)

        self.is_fit = True

    self.feature_memory = deque(maxlen=2)


def act(self, game_state: dict) -> str:
    if game_state['step'] == 1:
        self.feature_memory = reset_memory(self.feature_memory)

    features, self.feature_memory = state_to_features(game_state, self.feature_memory)
    if self.train:
        self.logger.debug(f"Current features: {features}")

    if not self.is_fit or np.random.rand() < self.exploration_rate:
        # explore
        action = np.random.choice(POSSIBLE_ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.logger.debug(f"Exploring random action, took action {action}")
        self.last_act_was_exploration = True
    else:
        # exploit
        q = self.q if not self.train else self.q_a_predict
        assert q is not None, "Playing does not work without weights!"

        actions = q.predict(features)[0]
        action = POSSIBLE_ACTIONS[np.argmax(actions)]
        self.logger.debug(f"Exploiting (predict actions), took action {action}")
        self.last_act_was_exploration = False

    return action


def state_to_features(game_state, feature_memory):
    all_features = []
    for features in feature_memory:
        all_features += features

    field = game_state['field'].T

    # field[field == 0] = (game_state['explosion_map'][field == 0] + 20)

    _, score, bombs_left, (self_x, self_y) = game_state['self']
    # others = game_state['others']

    field[self_x][self_y] = 5
    walls_and_crates_in_direction = [1 if field[self_x][self_y - 1] == -1 else 2 if field[self_x][self_y - 1] == 1 else 0,
                                     1 if field[self_x + 1][self_y] == -1 else 2 if field[self_x + 1][self_y] == 1 else 0,
                                     1 if field[self_x][self_y + 1] == -1 else 2 if field[self_x][self_y + 1] == 1 else 0,
                                     1 if field[self_x - 1][self_y] == -1 else 2 if field[self_x - 1][self_y] == 1 else 0,

                                     1 if field[self_x - 1][self_y - 1] == -1 else 2 if field[self_x - 1][self_y - 1] == 1 else 0,
                                     1 if field[self_x + 1][self_y + 1] == -1 else 2 if field[self_x + 1][self_y + 1] == 1 else 0,
                                     1 if field[self_x - 1][self_y + 1] == -1 else 2 if field[self_x - 1][self_y + 1] == 1 else 0,
                                     1 if field[self_x + 1][self_y - 1] == -1 else 2 if field[self_x + 1][self_y - 1] == 1 else 0,

                                     1 if self_y == 1 else 1 if field[self_x][self_y - 2] == -1 else 2 if field[self_x][self_y - 2] == 1 else 0,
                                     1 if self_x == 15 else 1 if field[self_x + 2][self_y] == -1 else 2 if field[self_x + 2][self_y] == 1 else 0,
                                     1 if self_y == 15 else 1 if field[self_x][self_y + 2] == -1 else 2 if field[self_x][self_y + 2] == 1 else 0,
                                     1 if self_x == 1 else 1 if field[self_x - 1][self_y] == -1 else 2 if field[self_x - 2][self_y] == 1 else 0
                                     ]

    [coin_x, coin_y], coin_dist = get_nearest_coin(game_state['coins'], (self_x, self_y))
    _, coin_dir = get_dir(coin_x, coin_y, self_x, self_y)
    # coin_dist_discrete = get_discrete_distance(coin_dist)

    [bomb_x, bomb_y], bomb_dist = get_nearest_bomb(game_state['bombs'], (self_x, self_y))
    _, bomb_dir = get_dir(bomb_x, bomb_y, self_x, self_y)
    # bomb_dist_discrete = get_discrete_distance(bomb_dist)

    [explosion_x, explosion_y], explosion_dist = get_nearest_explosion(game_state['explosion_map'], (self_x, self_y))
    _, explosion_dir = get_dir(explosion_x, explosion_y, self_x, self_y)
    # explosion_dist_discrete = get_discrete_distance(explosion_dist)

    [crate_x, crate_y], crate_dist = get_nearest_crate(field, (self_x, self_y))
    _, crate_dir = get_dir(crate_x, crate_y, self_x, self_y)
    # crate_dist_discrete = get_discrete_distance(crate_dist)

    can_lay_bomb = [1 if bombs_left else 0]

    # TODO: inside blast radius to nearest bomb (all bombs?)?

    current_features = walls_and_crates_in_direction + coin_dir + bomb_dir + explosion_dir + crate_dir + can_lay_bomb + [game_state['step'] / 100]
    feature_memory.append(current_features)

    all_features += current_features

    features = np.array(current_features)

    features = PolynomialFeatures(include_bias=True).fit_transform(features.reshape(1, -1))
    features = MinMaxScaler(copy=False).fit_transform(features.reshape(-1, 1))

    return features.reshape(1, -1), feature_memory


def get_nearest_coin(coin_map, self_pos):
    min_dist = None
    min_x, min_y = None, None

    for x, y in coin_map:
        distance_to_agent = np.abs(x - self_pos[0]) + np.abs(y - self_pos[1])
        if min_dist is None or distance_to_agent < min_dist:
            min_dist = distance_to_agent
            min_x, min_y = x, y

    return (min_x, min_y), min_dist


def get_dir(x, y, self_x, self_y):
    direction = [-1, -1, -1, -1]  # top, right, bottom, left

    if x is None or y is None:
        return False, direction

    pos_delta_x = x - self_x
    pos_delta_y = y - self_y

    # left or right
    if pos_delta_x > 0:
        direction[1] = pos_delta_x  # "RIGHT"
    elif pos_delta_x < 0:
        direction[3] = pos_delta_x   # "LEFT"

    # top or bottom
    if pos_delta_y > 0:
        direction[2] = pos_delta_y   # "DOWN"
    elif pos_delta_y < 0:
        direction[0] = pos_delta_y   # "UP"

    return True, direction


def get_nearest_crate(field, self_pos):
    min_dist = None
    min_x, min_y = None, None

    for [x, y], val in np.ndenumerate(field):
        if val != 1:
            continue

        distance_to_agent = np.abs(x - self_pos[0]) + np.abs(y - self_pos[1])
        if min_dist is None or distance_to_agent < min_dist:
            min_dist = distance_to_agent
            min_x, min_y = x, y

    return (min_x, min_y), min_dist


def get_nearest_bomb(bombs, self_pos):
    min_dist = None
    min_x, min_y = None, None

    for (x, y), _ in bombs:
        distance_to_agent = np.abs(x - self_pos[0]) + np.abs(y - self_pos[1])
        if min_dist is None or distance_to_agent < min_dist:
            min_dist = distance_to_agent
            min_x, min_y = x, y

    return (min_x, min_y), min_dist


def get_nearest_explosion(explosion_field, self_pos):
    min_dist = None
    min_x, min_y = None, None

    for [x, y], val in np.ndenumerate(explosion_field):
        if val == 0:
            continue

        distance_to_agent = np.abs(x - self_pos[0]) + np.abs(y - self_pos[1])
        if min_dist is None or distance_to_agent < min_dist:
            min_dist = distance_to_agent
            min_x, min_y = x, y

    return (min_x, min_y), min_dist


def reset_memory(memory):
    for i in range(memory.maxlen):
        memory.append(list(np.zeros(FEATURE_LENGTH//(memory.maxlen+1))))

    return memory

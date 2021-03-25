from datetime import datetime
import logging
import sys
import os

import numpy as np
import pickle

POSSIBLE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'weights.pt'

EXPLORATION_RATE = 0.7
MIN_EXPLORATION_RATE = 0.05
EXPLORATION_RATE_DECAY = 0.97

STATE_SPACE = 2**4*4*4


def setup(self):
    self.is_fit = False
    self.last_act_exploration = False

    if self.train:
        self.exploration_rate = EXPLORATION_RATE
    else:
        self.exploration_rate = MIN_EXPLORATION_RATE

    if not self.train:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    file_handler = logging.FileHandler(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    file_handler.setLevel(logging.DEBUG)
    self.logger.addHandler(file_handler)

    self.q = np.zeros((STATE_SPACE, len(POSSIBLE_ACTIONS)))

    if not self.train and os.path.exists(MODEL_FILE):
        self.logger.info("Using existing model to play")
        with open(MODEL_FILE, "rb") as file:
            self.q = pickle.load(file)

        self.is_fit = True


def act(self, game_state: dict) -> str:
    if not self.is_fit or np.random.rand() < self.exploration_rate:
        # explore
        self.logger.debug("Exploring random action")
        action = np.random.choice(POSSIBLE_ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.last_act_exploration = True
    else:
        # exploit
        q = self.q if not self.train else self.q_a
        self.logger.debug("Exploiting (predict actions)")
        action = POSSIBLE_ACTIONS[np.argmax(q[features_to_index(state_to_features(game_state)), :])]
        self.last_act_exploration = False

    self.exploration_rate *= EXPLORATION_RATE_DECAY
    self.exploration_rate = max(self.exploration_rate, 0.1)

    self.logger.debug(f"Took action {action}")
    return action


def state_to_features(game_state):
    field = game_state['field'].T

    # field[field == 0] = (game_state['explosion_map'][field == 0] + 20)

    _, score, bombs_left, (self_x, self_y) = game_state['self']
    # others = game_state['others']

    field[self_x][self_y] = 5
    walls_in_direction = [1 if field[self_x + 1][self_y] == -1 else 0,
                          1 if field[self_x - 1][self_y] == -1 else 0,
                          1 if field[self_x][self_y - 1] == -1 else 0,
                          1 if field[self_x][self_y + 1] == -1 else 0]

    # crate_in_direction = [1 if field[self_x + 1][self_y] == 1 else 0,
    #                       1 if field[self_x - 1][self_y] == 1 else 0,
    #                       1 if field[self_x][self_y - 1] == 1 else 0,
    #                       1 if field[self_x][self_y + 1] == 1 else 0]

    [coin_x, coin_y], coin_dist = get_nearest_coin(game_state['coins'], (self_x, self_y))
    coin_dir = get_dir(coin_x, coin_y, self_x, self_y)
    coin_dist_discrete = get_discrete_distance(coin_dist)

    # [crate_x, crate_y], crate_dist = get_nearest_crate(field, (self_x, self_y))
    # crate_dir = get_dir(crate_x, crate_y, self_x, self_y)
    # crate_dist_discrete = get_discrete_distance(crate_dist)

    return np.array(walls_in_direction + [coin_dir] + [coin_dist_discrete])


def get_nearest_coin(coin_map, self_pos):
    min_dist = 5000
    min_x, min_y = [-1, -1]
    for x, y in coin_map:
        distance_to_agent = np.abs(x - self_pos[0]) + np.abs(y - self_pos[1])
        if distance_to_agent < min_dist:
            min_dist = distance_to_agent
            min_x, min_y = x, y

    return (min_x, min_y), min_dist


def get_dir(x, y, self_x, self_y):
    pos_delta_x = x - self_x
    pos_delta_y = y - self_y

    if np.abs(pos_delta_x) >= np.abs(pos_delta_y):
        # left or right
        if pos_delta_x >= 0:
            return 1  # "RIGHT"
        else:
            return 0  # "LEFT"
    elif np.abs(pos_delta_x) < np.abs(pos_delta_y):
        # top or bottom
        if pos_delta_y >= 0:
            return 3  # "DOWN"
        else:
            return 2  # "UP"


def get_discrete_distance(dist):
    if dist > 10:
        dist_discrete = 3
    elif dist > 5:
        dist_discrete = 2
    elif dist > 2:
        dist_discrete = 1
    else:
        dist_discrete = 0

    return dist_discrete


# def get_nearest_crate(field, self_pos):
#     min_dist = 5000
#     min_x, min_y = [-1, -1]
#
#     for [x, y], val in np.ndenumerate(field):
#         if val != 1:
#             continue
#
#         distance_to_agent = np.abs(x - self_pos[0]) + np.abs(y - self_pos[1])
#         if distance_to_agent < min_dist:
#             min_dist = distance_to_agent
#             min_x, min_y = x, y
#
#     return (min_x, min_y), min_dist


def features_to_index(features):
    return 2**5*4**1*features[0] + 2**4*4**1*features[1] + 2**3*4**1*features[2] + 2**2*4**1*features[3] \
           + 4**1*features[4] + 4**0*features[5]

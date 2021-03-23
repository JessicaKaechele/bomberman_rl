from datetime import datetime
import logging
import sys
import os

import numpy as np
import pickle

POSSIBLE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'weights.pt'

EXPLORATION_RATE = 0.8
MIN_EXPLORATION_RATE = 0.1
EXPLORATION_RATE_DECAY = 0.99


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

    if not self.train and os.path.exists('a.pt'):
        self.logger.info("Using existing model to play")
        with open('a.pt', "rb") as file:
            self.model = pickle.load(file)

        self.is_fit = True


def act(self, game_state: dict) -> str:
    if not self.is_fit or np.random.rand() < self.exploration_rate:
        # explore
        self.logger.debug("Exploring random action")
        action = np.random.choice(POSSIBLE_ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.last_act_exploration = True
    else:
        # exploit
        model = self.model if not self.train else self.model_a
        self.logger.debug("Exploiting (predict actions)")
        action = POSSIBLE_ACTIONS[np.argmax(model.predict(state_to_features(game_state)))]
        self.last_act_exploration = False

    self.exploration_rate *= EXPLORATION_RATE_DECAY
    self.exploration_rate = max(self.exploration_rate, 0.1)

    self.logger.debug(f"Took action {action}")
    return action


def state_to_features(game_state):
    if game_state is None:
        return np.empty()

    field = game_state['field'].T

    # field[field == 0] = (game_state['explosion_map'][field == 0] + 20)

    _, score, bombs_left, (self_x, self_y) = game_state['self']
    # others = game_state['others']

    for coin_x, coin_y in game_state['coins']:
        field[coin_x][coin_y] = 50

    field[self_x][self_y] = 5
    walls_in_direction = [1 if field[self_x + 1][self_y] == -1 else 0,
                          1 if field[self_x - 1][self_y] == -1 else 0,
                          1 if field[self_x][self_y - 1] == -1 else 0,
                          1 if field[self_x][self_y + 1] == -1 else 0]

    coin_in_direction = [1 if field[self_x + 1][self_y] == 50 else 0,
                         1 if field[self_x - 1][self_y] == 50 else 0,
                         1 if field[self_x][self_y - 1] == 50 else 0,
                         1 if field[self_x][self_y + 1] == 50 else 0]

    [coin_x, coin_y], coin_dist = get_nearest_coin(game_state['coins'], (self_x, self_y))

    next_coin_dir = [0, 0, 0, 0]  # left, right, top, bottom
    pos_delta_x = coin_x - self_x
    pos_delta_y = coin_y - self_y

    if np.abs(pos_delta_x) >= np.abs(pos_delta_y):
        # left or right
        if pos_delta_x >= 0:
            next_coin_dir[1] = 1
        else:
            next_coin_dir[0] = 1
    elif np.abs(pos_delta_x) < np.abs(pos_delta_y):
        # top or bottom
        if pos_delta_y >= 0:
            next_coin_dir[3] = 1
        else:
            next_coin_dir[2] = 1


    # [1 if bombs_left else 0] +
    return np.array(walls_in_direction + coin_in_direction + [coin_dist] + next_coin_dir).reshape(1, -1)


def get_nearest_coin(coin_map, self_pos):
    min_dist = 5000
    min_x, min_y = [-1, -1]
    for x, y in coin_map:
        distance_to_agent = np.abs(x - self_pos[0]) + np.abs(y - self_pos[1])
        if distance_to_agent < min_dist:
            min_dist = distance_to_agent
            min_x, min_y = x, y

    return (min_x, min_y), min_dist

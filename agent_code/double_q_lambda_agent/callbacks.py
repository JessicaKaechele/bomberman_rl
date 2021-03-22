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

    if not self.train and os.path.exists(MODEL_FILE):
        self.logger.info("Using existing model to play")
        with open(MODEL_FILE, "rb") as file:
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

    field = game_state['field']
    _, score, bombs_left, (self_x, self_y) = game_state['self']
    # others = game_state['others']

    # explosion_map = game_state['explosion_map'
    coins = np.zeros_like(field)
    coins[tuple(np.array(game_state['coins']).T)] = 1

    features = np.concatenate((np.array(field).flatten(), np.array([score, 1 if bombs_left else 0, self_x, self_y]), np.array(coins).flatten()))

    return features.reshape(1, -1)

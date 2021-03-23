import os
import pickle
import random

import numpy as np
from sklearn.linear_model import SGDRegressor

from agent_code.my_function_approximation_agent_2.features import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = "jessi-saved-model.pt"


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # TODO: weights anders initialisieren
    if self.train or not os.path.isfile("jessi-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = []
        for _ in range(len(ACTIONS)):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([np.zeros(86)], [0])
            self.model.append(model)
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.model = pickle.load(file)
    self.epsilon = .1

    # if not self.train:
    #     handler = logging.StreamHandler(sys.stdout)
    #     handler.setLevel(logging.DEBUG)
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     handler.setFormatter(formatter)
    #     self.logger.addHandler(handler)
    #
    # file_handler = logging.FileHandler(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    # file_handler.setLevel(logging.DEBUG)
    # self.logger.addHandler(file_handler)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        self.logger.debug("Querying model for action.")
        features = state_to_features(game_state)
        #print(features)
        #print(np.array([m.predict([features])[0] for m in self.model]))
        action = ACTIONS[np.argmax(np.array([m.predict([features])[0] for m in self.model]))]
        print(action)
        return action
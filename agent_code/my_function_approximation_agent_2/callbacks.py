import os
import pickle
import random

import numpy as np
from sklearn.linear_model import SGDRegressor

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
            model.partial_fit([np.zeros(8)], [0])
            self.model.append(model)
    else:
        self.logger.info("Loading model from saved state.")
        with open("jessi-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    random_prob = .1
    # exploration:
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        self.logger.debug("Querying model for action.")
        features = state_to_features(game_state)
        action =  ACTIONS[np.argmax(np.array([m.predict([features])[0] for m in self.model]))]
        print(action)
        return action



def directions_to_nearest_coins(coins, x, y):
    directions = [0, 0, 0, 0]
    if coins:
        distances = np.sqrt((np.asarray(coins)[:, 0] - x) ** 2 + (np.asarray(coins)[:, 1] - y) ** 2)
        nearest_idx = np.argmin(distances)
        nearest_coin = coins[nearest_idx]
        if (nearest_coin[0] - x) < 0:
            directions[0] = 1
        elif (nearest_coin[0] - x) > 0:
            directions[1] = 1
        if (nearest_coin[1] - y) < 0:
            directions[2] = 1
        elif (nearest_coin[1] - y) > 0:
            directions[3] = 1
    return directions

def directions_to_wall(arena, x,y):
    directions = [1, 1, 1, 1]
    if arena[x+1,y] == -1:
        directions[0] = 0
    if arena[x-1, y] == -1:
        directions[1] = 0
    if arena[x, y+1] == -1:
        directions[2] = 0
    if arena[x, y-1] == -1:
        directions[3] = 0
    return directions


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']

    coin_map = np.zeros_like(arena)
    coin_map[tuple(np.array(coins).T)] = 1

    self_map = np.full(arena.shape, -1)
    self_map[x,y] = int(bombs_left)


    channels = []
    #channels.append(1)
    channels.append(directions_to_nearest_coins(coins, x, y))
    channels.append(directions_to_wall(arena, x, y))
    #channels.append(coin_map)
    #channels.append(self_map)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels).reshape(-1)
    # and return them as a vector
    return stacked_channels # np.array2string(stacked_channels.reshape(-1))

import os
import pickle
import random
from sklearn.multioutput import MultiOutputRegressor

from lightgbm import LGBMRegressor


import numpy as np

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
        #weights = np.random.rand(len(ACTIONS))
        self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
        self.is_fit = False
    else:
        self.logger.info("Loading model from saved state.")
        with open("jessi-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.is_fit = True


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # TODO: exploration anpassen
    random_prob = .1
    # exploration:
    if self.train and np.random.rand() < random_prob:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    if self.is_fit == True:
        q_values = self.model.predict(state_to_features(game_state))
    else:
        q_values = np.zeros(6).reshape(1, -1)
    return np.argmax(q_values[0])


def coin_ahead(new_pos, coins):
    if new_pos in coins:
        return 1
    else:
        return 0


def wall_ahead(new_pos, arena):
    if arena[new_pos] == -1:
        return 1
    else:
        return 0

def coin_forward(new_pos, old_pos, coins):

    try:
        if coins:
            distance_old = np.min(np.sqrt((np.asarray(coins)[:, 0] - old_pos[0]) ** 2 + (np.asarray(coins)[:, 1] - old_pos[1]) ** 2))
            distance_new = np.min(np.sqrt((np.asarray(coins)[:, 0] - new_pos[0]) ** 2 + (np.asarray(coins)[:, 1] - new_pos[1]) ** 2))
            if distance_new < distance_old:
                return 1
            else:
                return 0
        else:
            return 0
    except:
       print(new_pos, old_pos, coins)
def made_step(new_pos, old_pos):
    if new_pos != old_pos:
        return 1
    else:
        return 0

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
    # TODO: state to feature traversion
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

    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
         for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
             if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                 bomb_map[i, j] = min(bomb_map[i, j], t)

    coin_map = np.zeros_like(arena)
    coin_map[tuple(np.array(coins).T)] = 1

    enemy_map = np.zeros_like(arena)
    if others:
         enemy_map[tuple(np.array(others).T)] = 1

    self_map = np.full(arena.shape, -1)
    self_map[x,y] = int(bombs_left)



    channels = []
    #channels.append(arena)
    #channels.append(bomb_map)
    channels.append(coin_map)
    #channels.append(enemy_map)
    channels.append(self_map)
    #channels.append(explosion_map)
    #channels.append(nearest_distance)
    #channels.append(dir)
    #channels.append(x)
    #channels.append(y)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

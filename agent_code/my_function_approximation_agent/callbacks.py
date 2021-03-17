import os
import pickle
import random

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
        self.model = np.random.rand(5)
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
    # TODO: exploration anpassen
    random_prob = .1
    # exploration:
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        self.logger.debug("Querying model for action.")
        action_space = []
        for action in ACTIONS:
            action_space.append(state_to_features(game_state, action))
        Q = self.model * action_space
        Q = np.sum(Q, axis=1)
        return ACTIONS[np.argmax(Q)]


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

def state_to_features(game_state: dict, action: str) -> np.array:
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

    if action == 'UP':
        new_pos = (x,y - 1)
    elif action == 'DOWN':
        new_pos = (x, y + 1)
    elif action == 'RIGHT':
        new_pos = (x + 1, y)
    elif action == 'LEFT':
        new_pos = (x - 1, y)
    else:
        new_pos = (x,y)

    channels = []
    channels.append(1)
    channels.append(coin_ahead(new_pos, coins))
    channels.append(wall_ahead(new_pos, arena))
    channels.append(coin_forward(new_pos, (x,y), coins))
    channels.append(made_step(new_pos, (x,y)))
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return channels # np.array2string(stacked_channels.reshape(-1))

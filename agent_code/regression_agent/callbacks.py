import os
import pickle
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
    if self.train and not os.path.isfile("jessi-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
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
    random_prob = .3
    # exploration:
    if self.train and np.random.rand() < random_prob:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    if self.is_fit == True:
        q_values = self.model.predict([state_to_features(game_state)])
    else:
        q_values = np.zeros(6).reshape(1, -1)
    return ACTIONS[np.argmax(q_values[0])]


def coin_reachable(coin_directions, wall_directions):
    count_vals = dict(Counter(coin_directions))
    if 1 not in count_vals:
        return 0
    if count_vals[1] != 1:
        return 0
    return int(not (np.array(coin_directions) & np.array(wall_directions)).any())


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


def directions_to_wall(arena, x, y):
    directions = [0, 0, 0, 0]
    if arena[x - 1, y] == -1:
        directions[0] = 1
    if arena[x + 1, y] == -1:
        directions[1] = 1
    if arena[x, y - 1] == -1:
        directions[2] = 1
    if arena[x, y + 1] == -1:
        directions[3] = 1
    return directions


def crates_destroyable(arena, x, y):
    unique, counts = np.unique(arena[max(1, x - 3):min(15, x + 4), max(1, y - 3):min(15, y + 4)], return_counts=True)
    count_vals = dict(zip(unique, counts))
    if 1 in count_vals:
        return count_vals[1]
    return 0


def in_danger(field, bombs, x, y):
    arena = field.copy()
    arena[arena == -1] = 1

    directions = []
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < arena.shape[0]) and (0 < j < arena.shape[1]):
                arena[i, j] = 2  # min(arena[i, j], t)
            if (i, j) == (xb, yb):
                arena[i, j] = 3
    for x_i in range(x - 4, x + 5):
        for y_i in range(y - 4, y + 5):
            if x_i < 0 or x_i > 16 or y_i < 0 or y_i > 16:
                directions.append(1)
            else:
                directions.append(arena[x_i, y_i])
    # directions = [max(0,arena[x, y]), max(0,arena[x - 1, y]), max(0,arena[x + 1, y]), max(0,arena[x, y - 1]), max(0,arena[x, y + 1])]
    return directions


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector.

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
    coins = game_state['coins']

    channels = []

    # coin and self map
    # coin_map = np.zeros_like(arena)
    # coin_map[tuple(np.array(coins).T)] = 1

    # self_map = np.full(arena.shape, -1)
    # self_map[x, y] = int(bombs_left)
    # channels.append(coin_map)
    # channels.append(self_map)

    # central
    # central = get_central(arena, coins, x, y)
    # channels.append(central[10:20, 10:20])

    channels.append(directions_to_nearest_coins(coins, x, y))
    # channels.append(directions_to_wall(arena, x, y))
    stacked_channels = np.stack(channels).reshape(-1)

    crates = crates_destroyable(arena, x, y)
    danger = in_danger(arena, bombs, x, y)

    stacked_channels = np.append(stacked_channels, crates)
    stacked_channels = np.append(stacked_channels, np.array(danger))
    return stacked_channels


def get_central(arena, coins, x, y):
    central = np.full((34, 34), 0)
    x_new = 17 - x
    y_new = 17 - y
    central[y_new:(y_new + 17), x_new:(x_new + 17)] = arena.T
    coin_coords = np.array(coins).T
    coin_coords_x = y_new + coin_coords[0]
    coin_coords_y = x_new + coin_coords[1]
    central[(coin_coords_x, coin_coords_y)] = 1
    return central

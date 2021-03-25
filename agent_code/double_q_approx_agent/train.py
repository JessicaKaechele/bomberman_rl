import pickle
from collections import OrderedDict
from copy import deepcopy
from typing import List

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import events as e
from agent_code.double_q_approx_agent.callbacks import MODEL_FILE, POSSIBLE_ACTIONS
from agent_code.double_q_approx_agent.custom_events import POTENTIAL_TO_COLLECT_COIN, POTENTIAL_TO_DIE_BY_BOMB, \
    POTENTIAL_TO_DIE_BY_EXPLOSION
from agent_code.double_q_approx_agent.train_utils import fit_models_no_augment, fit_models_augmented_data, \
    events_from_state, LEARNING_RATE

ROUNDS_TO_UPDATE_MODEL = 200

USE_DATA_AUGMENTATION = True

GAME_REWARDS = {
    e.MOVED_LEFT: 0.001,
    e.MOVED_RIGHT: 0.001,
    e.MOVED_UP: 0.001,
    e.MOVED_DOWN: 0.001,
    e.WAITED: -0.01,
    e.INVALID_ACTION: -0.03,
    e.BOMB_EXPLODED: 0.002,
    e.BOMB_DROPPED: 0.0003,
    e.CRATE_DESTROYED: 0.05,
    e.COIN_FOUND: 0.06,
    e.COIN_COLLECTED: 0.4,
    e.KILLED_OPPONENT: 0.8,
    e.KILLED_SELF: -0.3,
    e.GOT_KILLED: -0.1,
    e.OPPONENT_ELIMINATED: 0,
    e.SURVIVED_ROUND: 0.1,
    POTENTIAL_TO_COLLECT_COIN: 0.01,
    POTENTIAL_TO_DIE_BY_BOMB: -0.01,
    POTENTIAL_TO_DIE_BY_EXPLOSION: -0.01
}

MODEL_FILE_TRAINING_A = 'weights_a.pt'
MODEL_FILE_TRAINING_B = 'weights_b.pt'
MODEL_FILE_TRAINING_TRACES = 'weights_traces.pt'


def setup_training(self):
    self.logger.debug("Training setup")

    self.q_a_learn = MultiOutputRegressor(SGDRegressor(eta0=LEARNING_RATE), n_jobs=-1)
    self.q_a_predict = MultiOutputRegressor(SGDRegressor(eta0=LEARNING_RATE), n_jobs=-1)

    self.q_b_learn = MultiOutputRegressor(SGDRegressor(eta0=LEARNING_RATE), n_jobs=-1)
    self.q_b_predict = MultiOutputRegressor(SGDRegressor(eta0=LEARNING_RATE), n_jobs=-1)

    self.q_a_learn.partial_fit([np.zeros(29)], [np.zeros(len(POSSIBLE_ACTIONS))])
    self.q_a_predict.partial_fit([np.zeros(29)], [np.zeros(len(POSSIBLE_ACTIONS))])
    self.q_b_learn.partial_fit([np.zeros(29)], [np.zeros(len(POSSIBLE_ACTIONS))])
    self.q_b_predict.partial_fit([np.zeros(29)], [np.zeros(len(POSSIBLE_ACTIONS))])

    self.learning_rate = LEARNING_RATE
    self.scaler = StandardScaler()

    self.round_count = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(
        f'DURING ROUND: Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self_action is None or old_game_state is None or new_game_state is None:
        return

    events = events_from_state(new_game_state, events)
    reward = reward_from_events(self, events)

    fit_models(self, old_game_state, self_action, new_game_state, reward, self.last_act_exploration)

    self.round_count += 1
    if self.round_count == ROUNDS_TO_UPDATE_MODEL:
        self.q_a_predict = None
        self.q_a_predict = deepcopy(self.q_a_learn)

        self.q_b_predict = None
        self.q_b_predict = deepcopy(self.q_b_learn)

        self.round_count = 0


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'END OF ROUND: Encountered event(s) {", ".join(map(repr, events))}')

    reward = reward_from_events(self, events)

    fit_models(self, last_game_state, last_action, None, reward, self.last_act_exploration)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.q_a_predict, file)


def reward_from_events(self, events: List[str]) -> int:
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def fit_models(self, old_game_state, action, new_game_state, reward, last_act_was_exploration):
    if USE_DATA_AUGMENTATION:
        fit_models_augmented_data(self, old_game_state, action, new_game_state, reward, last_act_was_exploration)
    else:
        fit_models_no_augment(self, old_game_state, action, new_game_state, reward, last_act_was_exploration)

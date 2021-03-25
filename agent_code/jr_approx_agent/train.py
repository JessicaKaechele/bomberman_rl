import pickle
from collections import OrderedDict
from copy import deepcopy
from typing import List

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import events as e
from agent_code.jr_approx_agent.callbacks import MODEL_FILE, POSSIBLE_ACTIONS
import agent_code.jr_approx_agent.custom_events as ce
from agent_code.jr_approx_agent.train_utils import fit_models_no_augment, fit_models_augmented_data, \
    events_from_state, LEARNING_RATE, USE_DOUBLE_Q

ROUNDS_TO_UPDATE_MODEL = 200

USE_DATA_AUGMENTATION = False

GAME_REWARDS = {
    e.MOVED_LEFT: 0,
    e.MOVED_RIGHT: 0,
    e.MOVED_UP: 0,
    e.MOVED_DOWN: 0,
    e.WAITED: -0.02,
    e.INVALID_ACTION: -0.06,
    e.BOMB_EXPLODED: 0.002,
    e.BOMB_DROPPED: 0.0003,
    e.CRATE_DESTROYED: 0.05,
    e.COIN_FOUND: 0.06,
    e.COIN_COLLECTED: 0.4,
    e.KILLED_OPPONENT: 0.8,
    e.KILLED_SELF: -0.3,
    e.GOT_KILLED: -0.1,
    e.OPPONENT_ELIMINATED: 0,
    e.SURVIVED_ROUND: 0.5,
    ce.POTENTIAL_TO_COLLECT_COIN: 0.01,
    ce.POTENTIAL_TO_NOT_COLLECT_COIN: -0.01,
    ce.POTENTIAL_TO_DIE_BY_BOMB: -0.01,
    ce.POTENTIAL_TO_NOT_DIE_BY_BOMB: 0.01,
    ce.POTENTIAL_TO_DIE_BY_EXPLOSION: -0.01,
    ce.POTENTIAL_TO_NOT_DIE_BY_EXPLOSION: 0.01,
    ce.POTENTIAL_TO_EXPLODE_CRATE: 0.01,
    ce.POTENTIAL_TO_NOT_EXPLODE_CRATE: -0.01
}

MODEL_FILE_TRAINING_A = 'weights_a.pt'
MODEL_FILE_TRAINING_B = 'weights_b.pt'
MODEL_FILE_TRAINING_TRACES = 'weights_traces.pt'


def setup_training(self):
    self.logger.debug("Training setup")

    feature_length = 253
    self.q_a_learn = MultiOutputRegressor(SGDRegressor(eta0=LEARNING_RATE), n_jobs=-1)
    self.q_a_predict = MultiOutputRegressor(SGDRegressor(eta0=LEARNING_RATE), n_jobs=-1)

    self.q_a_learn.partial_fit([np.zeros(feature_length)], [np.zeros(len(POSSIBLE_ACTIONS))])
    self.q_a_predict.partial_fit([np.zeros(feature_length)], [np.zeros(len(POSSIBLE_ACTIONS))])

    if USE_DOUBLE_Q:
        self.q_b_learn = MultiOutputRegressor(SGDRegressor(eta0=LEARNING_RATE), n_jobs=-1)
        self.q_b_predict = MultiOutputRegressor(SGDRegressor(eta0=LEARNING_RATE), n_jobs=-1)

        self.q_b_learn.partial_fit([np.zeros(feature_length)], [np.zeros(len(POSSIBLE_ACTIONS))])
        self.q_b_predict.partial_fit([np.zeros(feature_length)], [np.zeros(len(POSSIBLE_ACTIONS))])

    self.learning_rate = LEARNING_RATE

    self.eligibility_traces = {}
    for action in POSSIBLE_ACTIONS:
        self.eligibility_traces[action] = OrderedDict()

    self.round_count = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(
        f'DURING ROUND: Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self_action is None or old_game_state is None or new_game_state is None:
        return

    events = events_from_state(new_game_state, events)
    reward = reward_from_events(self, events)

    fit_models(self, old_game_state, self_action, new_game_state, reward)

    self.round_count += 1
    if self.round_count == ROUNDS_TO_UPDATE_MODEL:
        self.q_a_predict = None
        self.q_a_predict = deepcopy(self.q_a_learn)

        if USE_DOUBLE_Q:
            self.q_b_predict = None
            self.q_b_predict = deepcopy(self.q_b_learn)

        self.round_count = 0

        with open(MODEL_FILE, "wb") as file:
            pickle.dump(self.q_a_predict, file)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'END OF ROUND: Encountered event(s) {", ".join(map(repr, events))}')

    reward = reward_from_events(self, events)

    fit_models(self, last_game_state, last_action, None, reward)


def reward_from_events(self, events: List[str]) -> int:
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def fit_models(self, old_game_state, action, new_game_state, reward):
    if USE_DATA_AUGMENTATION:
        fit_models_augmented_data(self, old_game_state, action, new_game_state, reward)
    else:
        fit_models_no_augment(self, old_game_state, action, new_game_state, reward)

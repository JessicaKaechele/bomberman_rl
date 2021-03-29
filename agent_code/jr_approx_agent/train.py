import pickle
from collections import OrderedDict, deque
from copy import deepcopy
from typing import List

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

import events as e
from agent_code.jr_approx_agent.callbacks import MODEL_FILE, POSSIBLE_ACTIONS, \
    MIN_EXPLORATION_RATE, FEATURE_LENGTH
import agent_code.jr_approx_agent.custom_events as ce
from agent_code.jr_approx_agent.train_utils import _fit_models, \
    events_from_state, LEARNING_RATE, USE_DOUBLE_Q, MIN_LEARNING_RATE, LEARNING_RATE_DECAY

EVENTS_TO_UPDATE_MODEL = 500
ROUNDS_TO_UPDATE_LEARNING_RATE = 10
ROUNDS_TO_UPDATE_EXPLORATION_RATE = 5
EXPLORATION_RATE_DECAY = 0.99

GAME_REWARDS = {
    e.MOVED_LEFT: -0.0001,
    e.MOVED_RIGHT: -0.0001,
    e.MOVED_UP: -0.0001,
    e.MOVED_DOWN: -0.0001,
    e.WAITED: -0.1,
    e.INVALID_ACTION: -0.2,
    e.BOMB_EXPLODED: 0,
    e.BOMB_DROPPED: 0,
    e.CRATE_DESTROYED: 0,
    e.COIN_FOUND: 0,
    e.COIN_COLLECTED: 0.4,
    e.KILLED_OPPONENT: 0,  # 0.8,
    e.KILLED_SELF: 0,  # -0.7,
    e.GOT_KILLED: 0,  # -0.5,
    e.OPPONENT_ELIMINATED: 0,
    e.SURVIVED_ROUND: 0,  # 0.2,
    ce.POTENTIAL_TO_COLLECT_COIN: 0.01,
    ce.POTENTIAL_TO_NOT_COLLECT_COIN: -0.01,
    ce.POTENTIAL_TO_DIE_BY_BOMB: -0.04,
    ce.POTENTIAL_TO_NOT_DIE_BY_BOMB: 0.04,
    ce.POTENTIAL_TO_DIE_BY_EXPLOSION: -0.02,
    ce.POTENTIAL_TO_NOT_DIE_BY_EXPLOSION: 0.02,
    ce.POTENTIAL_TO_EXPLODE_CRATE: 0.001,
    ce.POTENTIAL_TO_NOT_EXPLODE_CRATE: -0.008,
    ce.DROPPED_BOMB_NEAR_CRATE: 0.25
}

MODEL_FILE_TRAINING_A = 'weights_a.pt'
MODEL_FILE_TRAINING_B = 'weights_b.pt'
MODEL_FILE_TRAINING_TRACES = 'weights_traces.pt'


def setup_training(self):
    self.logger.debug("Training setup")

    self.q_a_learn = MultiOutputRegressor(SGDRegressor(), n_jobs=-1)
    self.q_a_predict = MultiOutputRegressor(SGDRegressor(), n_jobs=-1)

    self.q_a_learn.partial_fit([np.zeros(FEATURE_LENGTH)], [np.zeros(len(POSSIBLE_ACTIONS))])
    self.q_a_predict.partial_fit([np.zeros(FEATURE_LENGTH)], [np.zeros(len(POSSIBLE_ACTIONS))])

    if USE_DOUBLE_Q:
        self.q_b_learn = MultiOutputRegressor(SGDRegressor(), n_jobs=-1)
        self.q_b_predict = MultiOutputRegressor(SGDRegressor(), n_jobs=-1)

        self.q_b_learn.partial_fit([np.zeros(FEATURE_LENGTH)], [np.zeros(len(POSSIBLE_ACTIONS))])
        self.q_b_predict.partial_fit([np.zeros(FEATURE_LENGTH)], [np.zeros(len(POSSIBLE_ACTIONS))])

    self.learning_rate = LEARNING_RATE

    self.eligibility_traces = {}
    for action in POSSIBLE_ACTIONS:
        self.eligibility_traces[action] = OrderedDict()

    self.round_count_weights = 0
    self.round_count_lr = 0
    self.round_count_er = 0

    # self.old_feature_memory = deque(maxlen=2)
    # self.new_feature_memory = deque(maxlen=2)
    # self.old_feature_memory = reset_memory(self.old_feature_memory)
    # self.new_feature_memory = reset_memory(self.new_feature_memory)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(
        f'DURING ROUND: Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self_action is None or old_game_state is None or new_game_state is None:
        return

    events = events_from_state(old_game_state, new_game_state, events)

    reward = reward_from_events(self, events)

    fit_models(self, old_game_state, self_action, new_game_state, reward)

    if self.round_count_weights == EVENTS_TO_UPDATE_MODEL:
        self.q_a_predict = None
        self.q_a_predict = deepcopy(self.q_a_learn)

        if USE_DOUBLE_Q:
            self.q_b_predict = None
            self.q_b_predict = deepcopy(self.q_b_learn)

        self.round_count_weights = 0

        with open(MODEL_FILE, "wb") as file:
            pickle.dump(self.q_a_predict, file)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'END OF ROUND: Encountered event(s) {", ".join(map(repr, events))}')

    reward = reward_from_events(self, events)

    fit_models(self, last_game_state, last_action, None, reward)

    self.round_count_weights += 1
    self.round_count_lr += 1
    self.round_count_er += 1

    if self.round_count_lr == ROUNDS_TO_UPDATE_LEARNING_RATE:
        self.learning_rate *= LEARNING_RATE_DECAY
        self.learning_rate = max(self.learning_rate, MIN_LEARNING_RATE)
        self.round_count_lr = 0
        self.logger.info(f"Current learning rate: {self.learning_rate}")

    if self.round_count_er == ROUNDS_TO_UPDATE_EXPLORATION_RATE:
        self.exploration_rate *= EXPLORATION_RATE_DECAY
        self.exploration_rate = max(self.exploration_rate, MIN_EXPLORATION_RATE)
        self.round_count_er = 0
        self.logger.info(f"Current exploration rate: {self.exploration_rate}")

    # self.old_feature_memory = reset_memory(self.old_feature_memory)
    # self.new_feature_memory = reset_memory(self.new_feature_memory)


def reward_from_events(self, events: List[str]) -> int:
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def fit_models(self, old_game_state, action, new_game_state, reward):
    _fit_models(self, old_game_state, action, new_game_state, reward)

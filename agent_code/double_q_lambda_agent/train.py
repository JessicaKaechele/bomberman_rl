import os
import pickle
import shutil
from collections import namedtuple, deque
from datetime import datetime
from typing import List

import numpy as np

import events as e
from agent_code.double_q_lambda_agent.callbacks import state_to_features, features_to_index
from agent_code.double_q_lambda_agent.callbacks import MODEL_FILE, POSSIBLE_ACTIONS, STATE_SPACE, MIN_EXPLORATION_RATE

LEARNING_RATE = 0.5
MIN_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.95
DISCOUNT = 0.8
LAMBDA = 0.5

DISTANCE_TO_COIN_GOT_SMALLER = 'DISTANCE_TO_COIN_GOT_SMALLER'
DISTANCE_TO_COIN_GOT_BIGGER = 'DISTANCE_TO_COIN_GOT_BIGGER'
GAME_REWARDS = {
    e.MOVED_LEFT: 0.0005,
    e.MOVED_RIGHT: 0.0005,
    e.MOVED_UP: 0.0005,
    e.MOVED_DOWN: 0.0005,
    e.WAITED: -0.001,
    e.INVALID_ACTION: -0.03,
    e.BOMB_EXPLODED: 0.002,
    e.BOMB_DROPPED: 0.0003,
    e.CRATE_DESTROYED: 0.05,
    e.COIN_FOUND: 0.06,
    e.COIN_COLLECTED: 0.3,
    e.KILLED_OPPONENT: 0.5,
    e.KILLED_SELF: -0.2,
    e.GOT_KILLED: -0.1,
    e.OPPONENT_ELIMINATED: 0,
    e.SURVIVED_ROUND: 0.1,
    DISTANCE_TO_COIN_GOT_SMALLER: 0.05,
    DISTANCE_TO_COIN_GOT_BIGGER: -0.055
}

MODEL_FILE_TRAINING_A = 'weights_a.pt'
MODEL_FILE_TRAINING_B = 'weights_b.pt'
MODEL_FILE_TRAINING_TRACES = 'weights_traces.pt'


def setup_training(self):
    self.logger.debug("Training setup")

    self.q_a = np.zeros((STATE_SPACE, len(POSSIBLE_ACTIONS)))
    self.q_b = np.zeros((STATE_SPACE, len(POSSIBLE_ACTIONS)))
    self.eligibility_traces = np.zeros((STATE_SPACE, len(POSSIBLE_ACTIONS)))
    self.learning_rate = LEARNING_RATE


def add_events_from_state(old_game_state, new_game_state, events):
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    if old_features[5] > new_features[5]:
        events.append(DISTANCE_TO_COIN_GOT_SMALLER)
    elif old_features[5] < new_features[5]:
        events.append(DISTANCE_TO_COIN_GOT_BIGGER)

    return events


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(
        f'DURING ROUND: Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self_action is None or old_game_state is None or new_game_state is None:
        return

    events = add_events_from_state(old_game_state, new_game_state, events)
    reward = reward_from_events(self, events)

    fit_models(self, old_game_state, self_action, new_game_state, reward, self.last_act_exploration)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'END OF ROUND: Encountered event(s) {", ".join(map(repr, events))}')

    reward = reward_from_events(self, events)

    fit_models(self, last_game_state, last_action, None, reward, self.last_act_exploration)

    self.learning_rate *= LEARNING_RATE_DECAY
    self.learning_rate = max(self.learning_rate, MIN_LEARNING_RATE)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.q_a, file)


def reward_from_events(self, events: List[str]) -> int:
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def fit_models(self, old_game_state, action, new_game_state, reward, last_act_was_exploration):
    old_state_idx = features_to_index(state_to_features(old_game_state))

    if self.is_fit:
        if new_game_state is not None:
            new_state_idx = features_to_index(state_to_features(new_game_state))

            model_a_new_q_values = self.q_a[new_state_idx, :]
            model_b_new_q_values = self.q_b[new_state_idx, :]

            model_a_old_q_value = self.q_a[old_state_idx, POSSIBLE_ACTIONS.index(action)]
            model_b_old_q_value = self.q_b[old_state_idx, POSSIBLE_ACTIONS.index(action)]

            q_update_a = self.learning_rate * (
                    reward + DISCOUNT * model_b_new_q_values[np.argmax(model_a_new_q_values)] - model_a_old_q_value)
            q_update_b = self.learning_rate * (
                    reward + DISCOUNT * model_a_new_q_values[np.argmax(model_b_new_q_values)] - model_b_old_q_value)
        else:
            model_a_old_q_value = self.q_a[old_state_idx, :]
            model_b_old_q_value = self.q_b[old_state_idx, :]

            q_update_a = self.learning_rate * (
                    reward - model_a_old_q_value[POSSIBLE_ACTIONS.index(action)])
            q_update_b = self.learning_rate * (
                    reward - model_b_old_q_value[POSSIBLE_ACTIONS.index(action)])
    else:
        q_update_a = self.learning_rate * reward
        q_update_b = self.learning_rate * reward

    self.eligibility_traces[old_state_idx, POSSIBLE_ACTIONS.index(action)] += 1

    for i in range(STATE_SPACE):
        for j in range(len(POSSIBLE_ACTIONS)):
            self.q_a[i, j] += (q_update_a * self.eligibility_traces[i, j])
            self.q_b[i, j] += (q_update_b * self.eligibility_traces[i, j])

            if last_act_was_exploration:
                self.eligibility_traces[i, j] = 0
            else:
                self.eligibility_traces[i, j] *= (DISCOUNT * LAMBDA)

    self.is_fit = True

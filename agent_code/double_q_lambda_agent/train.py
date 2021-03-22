import pickle
from collections import namedtuple
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, MODEL_FILE, POSSIBLE_ACTIONS

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'last_act_was_exploration'))

LEARNING_RATE = 0.5
LEARNING_RATE_DECAY = 0.95
DISCOUNT = 0.75
LAMBDA = 0.5

GAME_REWARDS = {
    e.MOVED_LEFT: 3,
    e.MOVED_RIGHT: 3,
    e.MOVED_UP: 3,
    e.MOVED_DOWN: 3,
    e.WAITED: -10,
    e.INVALID_ACTION: -50,
    e.BOMB_EXPLODED: 20,
    e.BOMB_DROPPED: 3,
    e.CRATE_DESTROYED: 2000,
    e.COIN_FOUND: 3000,
    e.COIN_COLLECTED: 5000,
    e.KILLED_OPPONENT: 15000,
    e.KILLED_SELF: -1000,
    e.GOT_KILLED: -500,
    e.OPPONENT_ELIMINATED: 200,
    e.SURVIVED_ROUND: 10000
}


def setup_training(self):
    self.logger.debug("Training setup")

    self.logger.info("Training mode")
    self.model_a = MultiOutputRegressor(LinearRegression(n_jobs=-1))
    self.model_b = MultiOutputRegressor(LinearRegression(n_jobs=-1))

    self.learning_rate = LEARNING_RATE

    self.transitions = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(
        f'DURING ROUND: Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self_action is None or old_game_state is None or new_game_state is None:
        return

    reward = reward_from_events(self, events)

    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward, self.last_act_exploration))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'END OF ROUND: Encountered event(s) {", ".join(map(repr, events))}')

    reward = reward_from_events(self, events)
    self.transitions.append(Transition(last_game_state, last_action, None, reward, self.last_act_exploration))

    fit_models(self)

    self.transitions = []

    self.learning_rate *= LEARNING_RATE_DECAY
    self.learning_rate = max(self.learning_rate, 0.1)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model_a, file)


def reward_from_events(self, events: List[str]) -> int:
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def fit_models(self):
    features = []
    targets_a = []
    targets_b = []
    eligibility_traces = np.zeros(len(POSSIBLE_ACTIONS))

    for idx, [previous_game_state, action, game_state, reward, last_act_was_exploration] in enumerate(self.transitions):
        old_state_features = state_to_features(previous_game_state)
        features.append(old_state_features.reshape(-1))

        if self.is_fit:
            if game_state is not None:
                new_state_features = state_to_features(game_state)

                model_a_new_q_values = self.model_a.predict(new_state_features)
                model_b_new_q_values = self.model_b.predict(new_state_features)

                model_a_old_q_value = self.model_a.predict(old_state_features)[0][POSSIBLE_ACTIONS.index(action)]
                model_b_old_q_value = self.model_b.predict(old_state_features)[0][POSSIBLE_ACTIONS.index(action)]

                q_update_a = self.learning_rate * (
                        reward + DISCOUNT * model_b_new_q_values[0][np.argmax(model_a_new_q_values)] - model_a_old_q_value)
                q_update_b = self.learning_rate * (
                        reward + DISCOUNT * model_a_new_q_values[0][np.argmax(model_b_new_q_values)] - model_b_old_q_value)
            else:
                model_a_new_q_values = self.model_a.predict(old_state_features)
                model_b_new_q_values = self.model_b.predict(old_state_features)

                q_update_a = self.learning_rate * (
                        reward - model_a_new_q_values[0][POSSIBLE_ACTIONS.index(action)])
                q_update_b = self.learning_rate * (
                        reward - model_b_new_q_values[0][POSSIBLE_ACTIONS.index(action)])
        else:
            model_a_new_q_values = np.zeros(len(POSSIBLE_ACTIONS)).reshape(1, -1)
            model_b_new_q_values = np.zeros(len(POSSIBLE_ACTIONS)).reshape(1, -1)

            q_update_a = self.learning_rate * reward
            q_update_b = self.learning_rate * reward

        eligibility_traces[POSSIBLE_ACTIONS.index(action)] += 1

        for i in range(len(POSSIBLE_ACTIONS)):
            model_a_new_q_values[0][i] += (q_update_a * eligibility_traces[i])
            model_b_new_q_values[0][i] += (q_update_b * eligibility_traces[i])

            if last_act_was_exploration:
                eligibility_traces[i] = 0
            else:
                eligibility_traces[i] *= (DISCOUNT * LAMBDA)

        targets_a.append(model_a_new_q_values[0])
        targets_b.append(model_b_new_q_values[0])

    self.model_a.fit(features, normalize(targets_a))
    self.model_b.fit(features, normalize(targets_b))
    self.is_fit = True

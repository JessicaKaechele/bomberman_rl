import numpy as np

from agent_code.jr_approx_agent.callbacks import POSSIBLE_ACTIONS, state_to_features, get_nearest_coin, \
    get_nearest_bomb, get_nearest_explosion, get_nearest_crate
import agent_code.jr_approx_agent.custom_events as ce
import events as e

LEARNING_RATE = 0.5
MIN_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.995
DISCOUNT = 0.8
LAMBDA = 0.85

USE_DOUBLE_Q = False
KEEP_N_ELIGIBILITY_TRACES_PER_ACTION = 100


def events_from_state(old_game_state, new_game_state, events):
    if new_game_state is None:
        return events

    _, _, _, old_pos = old_game_state['self']
    _, _, _, new_pos = new_game_state['self']

    def coin_event(current_events):
        _, distance = get_nearest_coin(new_game_state['coins'], new_pos)

        if distance is not None:
            if distance >= 4:
                distance_multiplier = min(distance - 3, 3)
                current_events += list(np.repeat(ce.POTENTIAL_TO_NOT_COLLECT_COIN, distance_multiplier))
            else:
                distance_multiplier = min(4 - distance, 3)
                current_events += list(np.repeat(ce.POTENTIAL_TO_COLLECT_COIN, distance_multiplier))

        return current_events

    def bomb_event(current_events):
        [bomb_x, bomb_y], distance = get_nearest_bomb(new_game_state['bombs'], new_pos)

        if distance is not None:
            if bomb_x != new_pos[0] and bomb_y != new_pos[1]:
                current_events += list(np.repeat(ce.POTENTIAL_TO_NOT_DIE_BY_BOMB, 3))
            elif distance >= 4:
                distance_multiplier = min(distance - 3, 3)
                current_events += list(np.repeat(ce.POTENTIAL_TO_NOT_DIE_BY_BOMB, distance_multiplier))
            else:
                current_events += list(np.repeat(ce.POTENTIAL_TO_DIE_BY_BOMB, 3))

        return current_events

    def explosion_event(current_events):
        _, distance = get_nearest_explosion(new_game_state['explosion_map'], new_pos)

        if distance is not None:
            if distance >= 2:
                distance_multiplier = min(distance - 1, 3)
                current_events += list(np.repeat(ce.POTENTIAL_TO_NOT_DIE_BY_EXPLOSION, distance_multiplier))
            else:
                distance_multiplier = min(2 - distance, 3)
                current_events += list(np.repeat(ce.POTENTIAL_TO_DIE_BY_EXPLOSION, distance_multiplier))

        return current_events

    def crate_events(current_events):
        old_create_pos, old_distance = get_nearest_crate(new_game_state['field'], old_pos)
        create_pos, distance = get_nearest_crate(new_game_state['field'], new_pos)

        if old_distance is not None and distance is not None:
            if old_distance < distance:
                current_events += list(np.repeat(ce.POTENTIAL_TO_NOT_EXPLODE_CRATE, 3))
            elif old_distance > distance:
                current_events += list(np.repeat(ce.POTENTIAL_TO_EXPLODE_CRATE, 3))

        if distance is not None:
            if distance >= 4:
                distance_multiplier = min(distance - 2, 3)
                current_events += list(np.repeat(ce.POTENTIAL_TO_NOT_EXPLODE_CRATE, distance_multiplier))
            else:
                distance_multiplier = min(3 - distance, 3)
                current_events += list(np.repeat(ce.POTENTIAL_TO_EXPLODE_CRATE, distance_multiplier))

            if distance == 1 and e.BOMB_DROPPED in current_events:
                current_events += [ce.DROPPED_BOMB_NEAR_CRATE]

        return current_events

    events = coin_event(events)
    events = bomb_event(events)
    events = explosion_event(events)
    events = crate_events(events)

    return events


def _fit_models(self, old_game_state, action, new_game_state, reward):
    old_features = state_to_features(self, old_game_state)
    new_features = None if new_game_state is None else state_to_features(self, new_game_state)

    self.eligibility_traces[action] = update_eligibility_trace_dict(
        self.eligibility_traces[action], hash(str(old_features.tostring())), old_features)

    if USE_DOUBLE_Q:
        fit_double_q(self, old_features, action, new_features, reward)
    else:
        fit_q(self, old_features, action, new_features, reward)


def fit_double_q(self, old_game_features, action, new_game_features, reward):
    def fit(q_a_predict, q_a_learn, q_b_predict, action_param):
        model_a_old_q_value = q_a_predict.predict(old_game_features)[0][POSSIBLE_ACTIONS.index(action_param)]

        if self.is_fit and new_game_features is not None:
            model_a_new_q_values = q_a_predict.predict(new_game_features)[0]
            model_b_new_q_values = q_b_predict.predict(new_game_features)[0]

            q_update_a = self.learning_rate * (
                    reward + DISCOUNT * model_b_new_q_values[np.argmax(model_a_new_q_values)] - model_a_old_q_value)
        else:
            q_update_a = self.learning_rate * reward

        q_a_learn.estimators_[POSSIBLE_ACTIONS.index(action)].partial_fit(old_game_features,
                                                                               [model_a_old_q_value + q_update_a])

        # for action_idx, action_value in enumerate(POSSIBLE_ACTIONS):
        #     trace_dict = self.eligibility_traces[action_value]
        #     for key, value in trace_dict.items():
        #         features = value['features']
        #         eligibility_value = value['eligibility_value']
        #
        #         q_value = q_a_predict.estimators_[POSSIBLE_ACTIONS.index(action_value)].predict(features)
        #         target_q_a = q_value + q_update_a * eligibility_value
        #
        #         q_a_learn.estimators_[POSSIBLE_ACTIONS.index(action_value)].partial_fit(features, target_q_a)
        #
        #         if not self.last_act_was_exploration:
        #             self.eligibility_traces[action_value][key] = {'features': features,
        #                                                           'eligibility_value': eligibility_value * LAMBDA}
        #         else:
        #             self.eligibility_traces[action_value][key] = {'features': features,
        #                                                           'eligibility_value': 0}

        return q_a_predict, q_a_learn, q_b_predict

    if np.random.rand() >= 0.5:
        # update model a
        self.q_a_predict, self.q_a_learn, self.q_b_predict = fit(self.q_a_predict, self.q_a_learn, self.q_b_predict, action)
    else:
        # update model b
        self.q_b_predict, self.q_b_learn, self.q_a_predict = fit(self.q_b_predict, self.q_b_learn, self.q_a_predict, action)

    self.is_fit = True


def fit_q(self, old_game_features, action, new_game_features, reward):
    model_a_old_q_value = self.q_a_predict.predict(old_game_features)[0][POSSIBLE_ACTIONS.index(action)]
    self.logger.debug(f"Prediction for old game state: {model_a_old_q_value}")

    if self.is_fit and new_game_features is not None:
        model_a_new_q_values = self.q_a_predict.predict(new_game_features)[0]
        self.logger.debug(f"Prediction for new game state: {model_a_new_q_values}")

        q_update_a = self.learning_rate * (
                reward + DISCOUNT * model_a_new_q_values[np.argmax(model_a_new_q_values)] - model_a_old_q_value)
    else:
        q_update_a = self.learning_rate * reward

    # for no eligibility traces:
    self.q_a_learn.estimators_[POSSIBLE_ACTIONS.index(action)].partial_fit(old_game_features,
                                                                           [model_a_old_q_value + q_update_a])

    self.logger.debug(f"Current q_update value: {q_update_a}")

    # for action_idx, action_value in enumerate(POSSIBLE_ACTIONS):
    #     trace_dict = self.eligibility_traces[action_value]
    #     for key, value in trace_dict.items():
    #         features = value['features']
    #         eligibility_value = value['eligibility_value']
    #
    #         q_value = self.q_a_predict.estimators_[POSSIBLE_ACTIONS.index(action_value)].predict(features)
    #         target_q_a = q_value + q_update_a * eligibility_value
    #
    #         self.q_a_learn.estimators_[POSSIBLE_ACTIONS.index(action_value)].partial_fit(features, target_q_a)
    #
    #         if not self.last_act_was_exploration:
    #             self.eligibility_traces[action_value][key] = {'features': features,
    #                                                           'eligibility_value': eligibility_value * LAMBDA}
    #         else:
    #             self.eligibility_traces[action_value][key] = {'features': features,
    #                                                           'eligibility_value': 0}

    self.is_fit = True


def update_eligibility_trace_dict(eligibility_trace_dict, state_key, features):
    # if state_key in eligibility_trace_dict:
    #     # existing key, move to front and increment
    #     eligibility_trace_dict.move_to_end(state_key)
    #     current = eligibility_trace_dict[state_key]
    #     eligibility_trace_dict[state_key] = {'features': features,
    #                                          'eligibility_value': current['eligibility_value'] + 1}
    # else:
    #     # new key, remove oldest item if max size was reached and insert new key
    #     while len(eligibility_trace_dict) >= KEEP_N_ELIGIBILITY_TRACES_PER_ACTION:
    #         eligibility_trace_dict.popitem(False)
    #
    #     eligibility_trace_dict[state_key] = {'features': features, 'eligibility_value': 1}

    return eligibility_trace_dict

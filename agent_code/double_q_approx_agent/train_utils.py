import numpy as np

from agent_code.double_q_approx_agent.callbacks import POSSIBLE_ACTIONS, state_to_features, get_nearest_coin, \
    get_discrete_distance, get_nearest_bomb, get_nearest_explosion
from agent_code.double_q_approx_agent.custom_events import POTENTIAL_TO_COLLECT_COIN, POTENTIAL_TO_DIE_BY_BOMB, \
    POTENTIAL_TO_DIE_BY_EXPLOSION

LEARNING_RATE = 0.5
MIN_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.95
DISCOUNT = 0.8
LAMBDA = 0.5


def feature_augmentation(features):
    # top, right, bottom, left
    def rotate(feature_to_rotate):
        result = np.empty_like(feature_to_rotate)
        result[0] = feature_to_rotate[3]
        result[1] = feature_to_rotate[0]
        result[2] = feature_to_rotate[1]
        result[3] = feature_to_rotate[2]
        return result

    def mirror(feature_to_mirror):
        result = np.empty_like(feature_to_mirror)
        result[0] = feature_to_mirror[2]
        result[1] = feature_to_mirror[3]
        result[2] = feature_to_mirror[0]
        result[3] = feature_to_mirror[1]
        return result

    # mirror and rotate features, call before fit to get 8x the data!
    all_augmentations = np.empty((8, features.shape[0]))

    rotated_wall_dir = features[0:4]
    rotated_coin_dir = features[4:8]
    rotated_bomb_dir = features[8:12]
    rotated_explosion_dir = features[12:16]
    for i in range(4):
        rotated_wall_dir = rotate(rotated_wall_dir)
        mirrored_wall_dir = mirror(rotated_wall_dir)
        all_augmentations[i * 2 + 0][0:4] = rotated_wall_dir
        all_augmentations[i * 2 + 1][0:4] = mirrored_wall_dir

        rotated_coin_dir = rotate(rotated_coin_dir)
        mirrored_coin_dir = mirror(rotated_coin_dir)
        all_augmentations[i * 2 + 0][4:8] = rotated_coin_dir
        all_augmentations[i * 2 + 1][4:8] = mirrored_coin_dir

        rotated_bomb_dir = rotate(rotated_bomb_dir)
        mirrored_bomb_dir = mirror(rotated_bomb_dir)
        all_augmentations[i * 2 + 0][8:12] = rotated_bomb_dir
        all_augmentations[i * 2 + 1][8:12] = mirrored_bomb_dir

        rotated_explosion_dir = rotate(rotated_explosion_dir)
        mirrored_explosion_dir = mirror(rotated_explosion_dir)
        all_augmentations[i * 2 + 0][12:16] = rotated_explosion_dir
        all_augmentations[i * 2 + 1][12:16] = mirrored_explosion_dir

        all_augmentations[i * 2 + 0][16:] = features[16:]
        all_augmentations[i * 2 + 1][16:] = features[16:]

    return all_augmentations


def action_augmentation(action):
    # top, right, bottom, left
    def rotate(action_to_rotate):
        if action_to_rotate == "UP":
            return "RIGHT"
        elif action_to_rotate == "RIGHT":
            return "DOWN"
        elif action_to_rotate == "DOWN":
            return "LEFT"
        elif action_to_rotate == "LEFT":
            return "UP"

    def mirror(action_to_mirror):
        if action_to_mirror == "UP":
            return "DOWN"
        elif action_to_mirror == "RIGHT":
            return "LEFT"
        elif action_to_mirror == "DOWN":
            return "UP"
        elif action_to_mirror == "LEFT":
            return "RIGHT"

    # mirror and rotate features, call before fit to get 8x the data!
    all_augmentations = np.empty((8, 1), dtype=object)
    if action == "WAIT" or action == "BOMB":
        all_augmentations[:] = action
    else:
        rotated_action = action
        for i in range(4):
            rotated_action = rotate(rotated_action)
            mirrored_action = mirror(rotated_action)
            all_augmentations[i * 2 + 0] = rotated_action
            all_augmentations[i * 2 + 1] = mirrored_action

    return all_augmentations


def events_from_state(new_game_state, events):
    if new_game_state is None:
        return events

    _, _, _, new_pos = new_game_state['self']
    distance_count = np.array([4, 3, 2, 1])

    def coin_event(current_events):
        _, new_dist = get_nearest_coin(new_game_state['coins'], new_pos)
        discrete_dist = get_discrete_distance(new_dist)

        current_events += list(np.repeat(POTENTIAL_TO_COLLECT_COIN, np.multiply(distance_count, discrete_dist).sum()))

        return current_events

    def bomb_event(current_events):
        _, new_dist = get_nearest_bomb(new_game_state['bombs'], new_pos)
        discrete_dist = get_discrete_distance(new_dist)

        current_events += list(np.repeat(POTENTIAL_TO_DIE_BY_BOMB, np.multiply(distance_count, discrete_dist).sum()))

        return current_events

    def explosion_event(current_events):
        _, new_dist = get_nearest_explosion(new_game_state['explosion_map'], new_pos)
        discrete_dist = get_discrete_distance(new_dist)

        current_events += list(np.repeat(POTENTIAL_TO_DIE_BY_EXPLOSION, np.multiply(distance_count, discrete_dist).sum()))

        return current_events

    events = coin_event(events)
    events = bomb_event(events)
    events = explosion_event(events)

    return events


def fit_models_augmented_data(self, old_game_state, action, new_game_state, reward, last_act_was_exploration):
    old_features = state_to_features(old_game_state)
    new_features = None if new_game_state is None else state_to_features(new_game_state)

    augmented_old = feature_augmentation(old_features)
    augmented_new = None if new_features is None else feature_augmentation(new_features)

    augmented_action = action_augmentation(action)

    for idx, old_augmented_feature_instance in enumerate(augmented_old):
        old_features_reshaped = old_augmented_feature_instance.reshape(1, -1)
        action_instance = augmented_action[idx][0]
        new_features_reshaped = None if augmented_new is None else augmented_new[idx].reshape(1, -1)

        fit(self, old_features_reshaped, action_instance, new_features_reshaped, reward, last_act_was_exploration)


def fit_models_no_augment(self, old_game_state, action, new_game_state, reward, last_act_was_exploration):
    old_features = state_to_features(old_game_state)
    new_features = None if new_game_state is None else state_to_features(new_game_state)

    old_features_reshaped = old_features.reshape(1, -1)
    new_features_reshaped = None if new_features is None else new_features.reshape(-1, 1)

    fit(self, old_features_reshaped, action, new_features_reshaped, reward, last_act_was_exploration)


def fit(self, old_game_features, action, new_game_features, reward, last_act_was_exploration):
    model_a_old_q_value = self.q_a_predict.predict(old_game_features)[0][POSSIBLE_ACTIONS.index(action)]
    model_b_old_q_value = self.q_b_predict.predict(old_game_features)[0][POSSIBLE_ACTIONS.index(action)]

    if self.is_fit and new_game_features is not None:
        model_a_new_q_values = self.q_a_predict.predict(new_game_features)[0]
        model_b_new_q_values = self.q_b_predict.predict(new_game_features)[0]

        q_update_a = self.learning_rate * (
                reward + DISCOUNT * model_b_new_q_values[np.argmax(model_a_new_q_values)] - model_a_old_q_value)
        q_update_b = self.learning_rate * (
                reward + DISCOUNT * model_a_new_q_values[np.argmax(model_b_new_q_values)] - model_b_old_q_value)
    else:
        q_update_a = self.learning_rate * reward
        q_update_b = self.learning_rate * reward

    self.q_a_learn.estimators_[POSSIBLE_ACTIONS.index(action)].partial_fit(old_game_features, [model_a_old_q_value + q_update_a])
    self.q_b_learn.estimators_[POSSIBLE_ACTIONS.index(action)].partial_fit(old_game_features, [model_b_old_q_value + q_update_b])

    self.is_fit = True

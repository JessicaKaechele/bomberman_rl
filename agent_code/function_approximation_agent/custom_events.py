import numpy as np

from agent_code.function_approximation_agent.features import coin_reachable, directions_to_nearest_coins, \
    directions_to_wall, crates_destroyable

DISTANCE_2 = "DISTANCE_2"
DISTANCE_1 = "DISTANCE_1"
DISTANCE_0 = "DISTANCE_0"
COIN_NOT_REACHABLE = "COIN_NOT_REACHABLE"
DROP_BOMB_NEAR_CRATE = "DROP_BOMB_NEAR_CRATE"
CAN_NOT_ESCAPE = "CAN_NOT_ESCAPE"
DID_NOT_ESCAPE = "DID_NOT_ESCAPE"

def not_moving_event(x,y, events, self):
    if len(self.last_positions) == 5:
        dis_2 = np.sqrt(
            (self.last_positions[2][0] - x) ** 2 + (self.last_positions[2][1] - y) ** 2)
        dis_1 = np.sqrt(
            (self.last_positions[1][0] - x) ** 2 + (self.last_positions[1][1] - y) ** 2)
        dis_0 = np.sqrt(
            (self.last_positions[0][0] - x) ** 2 + (self.last_positions[0][1] - y) ** 2)
        if dis_0 < 2:
            events.append(DISTANCE_0)
        elif dis_1 < 2:
            events.append(DISTANCE_1)
        elif dis_2 < 2:
            events.append(DISTANCE_2)


    self.last_positions.append((x,y))

def did_not_escape_event(self, x, y, events):
    if "BOMB_DROPPED" in events and self.dropped_bomb == False:
        self.dropped_bomb = True
        self.bomb_counter = 4
        self.bomb_position = (x, y)
        self.bomb_distance = 0
        return
    if self.dropped_bomb and self.bomb_counter > 0:
        distance = np.sqrt((x-self.bomb_position[0])**2 + (y-self.bomb_position[1])**2)
        self.bomb_counter -= 1
        if distance > self.bomb_distance:
            events.append(DID_NOT_ESCAPE)
            self.bomb_distance = distance
        elif (x != self.bomb_position[0] and y != self.bomb_position[1]):
            events.append(DID_NOT_ESCAPE)

    else:
        self.dropped_bomb = False

def can_not_escape_event(events, new_game_state, x, y):
    if "BOMB_DROPPED" in events and can_not_escape(new_game_state, x, y):
        events.append(CAN_NOT_ESCAPE)


def get_next_steps(field, x, y, step_nr):
    idxs = [[x-step_nr, x+step_nr, x, x],[y,y, y-step_nr, y+step_nr]]
    danger_fields = field[tuple(idxs)]
    return danger_fields, [np.array(idxs[0])[danger_fields == 0],np.array(idxs[1])[danger_fields == 0]]


def can_not_escape(game_state, x,y):
    arena = game_state['field']
    arena = arena.copy()
    arena[arena == -1] = 1
    field = []
    for x_i in range(x-4, x+5):
        for y_i in range(y-4, y+5):
            if x_i < 0 or x_i > 16 or y_i < 0 or y_i > 16:
                field.append(1)
            else:
                field.append(arena[x_i, y_i])

    field = np.array(field).reshape((9,9))
    middle = int(len(field) / 2)

    first_step, idxs = get_next_steps(field, middle, middle, 1)
    if not 0 in first_step: # escape is not possible because all first steps are blocked
        return True
    second_steps = []
    second_idxs_x = np.array([])
    second_idxs_y = np.array([])
    for i in range(len(idxs[0])):
        second_step, second_idx = get_next_steps(field, idxs[0][i], idxs[1][i], 1)
        second_steps.append(second_step)
        second_idxs_x = np.concatenate((second_idxs_x, second_idx[0]))
        second_idxs_y = np.concatenate((second_idxs_y, second_idx[1]))

    if not 0 in np.array(second_steps): # escape is not possible because all second steps are blocked
        return True

    third_steps = []
    third_idxs_x = np.array([])
    third_idxs_y = np.array([])
    for i in range(len(second_idxs_x)):
        third_step, third_idx = get_next_steps(field, int(second_idxs_x[i]), int(second_idxs_y[i]), 1)
        third_steps.append(third_step)
        third_idxs_x = np.concatenate((third_idxs_x, third_idx[0]))
        third_idxs_y = np.concatenate((third_idxs_y, third_idx[1]))

    if not 0 in np.array(third_steps): # escape is not possible because all third steps are blocked
        return True

    fourth_steps = []
    for i in range(len(second_idxs_x)):
        fourth_step, fourth_idx = get_next_steps(field, int(third_idxs_x[i]), int(third_idxs_y[i]), 1)
        fourth_steps.append(fourth_step)

    if not 0 in np.array(fourth_steps): # escape is not possible because all fourth steps are blocked
        return True

    return False

def bomb_near_crate_event(arena, events, x, y):
    if "BOMB_DROPPED" in events and crates_destroyable(arena, x, y) > 0:
        events.append(DROP_BOMB_NEAR_CRATE)


def coin_reachable_event(arena, coins, events, x, y):
    if not coin_reachable(directions_to_nearest_coins(coins, x, y), directions_to_wall(arena, x, y)):
        events.append(COIN_NOT_REACHABLE)
import pickle
from .callbacks import MODEL_FILE

def add_statistics(self, events):
    for event in events:
        if event == "COIN_COLLECTED":
            self.collected_coins_episode += 1
        elif event == "CRATE_DESTROYED":
            self.destroyed_crates_episode += 1
        elif event == "BOMB_DROPPED":
            self.dropped_bombs += 1


def end_statistics(self, events):
    add_statistics(self, events)

    with open("statistics/coins_collected", "a") as f:
        f.writelines(str(self.collected_coins_episode)+ "\n")
    with open("statistics/crates_destroyed", "a") as f:
        f.writelines(str(self.destroyed_crates_episode)+ "\n")
    with open("statistics/dropped_bombs", "a") as f:
        f.writelines(str(self.dropped_bombs) + "\n")
    self.dropped_bombs = 0
    self.collected_coins_episode = 0
    self.destroyed_crates_episode = 0


def save_model(self):
    # Store the model
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model, file)


def save_rewards(self):
    with open("rewards/statistics", "a") as f:
        for reward in self.all_rewards:
            f.writelines(str(reward) + "\n")
    self.all_rewards = []
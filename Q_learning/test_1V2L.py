import gymnasium as gym
import highway_env
import numpy as np
import random
import time

def is_in_visited_states(state, visited_states, threshold=0.0001):
    for i, visited_state in enumerate(visited_states):
        if np.linalg.norm(np.array(state) - np.array(visited_state)) < threshold:
            return True, i  # Return index if found
    return False, None


# Create the environment
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "lanes_count": 2,
        "vehicles_count": 2,
        "initial_lane_id": 0
    },
        "order": "sorted",
        "lanes_count": 2,
        "vehicles_count": 2,
        "collision_reward": -20,#-10
        "high_speed_reward": 0,#0
        "lane_change_reward": 0,#0
        "right_lane_reward": 0,
        "on_road_reward": -20,
        "initial_lane_id": 0,
        "absolute": False
}
env = gym.make('highway-v0', render_mode="human", config=config)
# List to store total rewards and actions for each episode
state_policy = []
state_V = []
total_rewards = []
action_length = []
visited_states = []

import pickle
with open("state_policy.pkl", "rb") as f:
    state_policy = pickle.load(f)
with open("state_V.pkl", "rb") as f:
    state_V = pickle.load(f)
with open("total_rewards.pkl", "rb") as f:
    total_rewards = pickle.load(f)
with open("action_length.pkl", "rb") as f:
    action_length = pickle.load(f)
with open("visited_states.pkl", "rb") as f:
    visited_states = pickle.load(f)
test_action_length = 0
reward1 = 0
state, _ = env.reset()
done = False

while not done:
    s1 = np.round(state,2)
    print("state:", s1)
    c1, c2 = is_in_visited_states(s1,visited_states)
    if c1:
      action = state_policy[c2]
      print("|",c2,"|")
    else:
      action = env.action_space.sample()
      print("?")
    n_state, reward, done, additional_info, info = env.step(action)
    test_action_length += 1
    print(action, reward)
    a = n_state
    # print(a)
    reward1 += reward
    time.sleep(0.5)
    state = n_state
    if additional_info:
      done = True
print(test_action_length,reward1)

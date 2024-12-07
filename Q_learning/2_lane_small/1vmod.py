import gymnasium as gym
import highway_env
import numpy as np
import random
import time
from itertools import product
from scipy.spatial import KDTree

state_policy = []
state_V = []
total_rewards = []
action_length = []
states = []
visited_time = []

import pickle
with open("state_policy.pkl", "rb") as f:
    state_policy = pickle.load(f)
with open("states.pkl", "rb") as f:
    states = pickle.load(f)
with open("state_V.pkl", "rb") as f:
    state_V = pickle.load(f)
with open("total_rewards.pkl", "rb") as f:
    total_rewards = pickle.load(f)
with open("action_length.pkl", "rb") as f:
    action_length = pickle.load(f)
with open("visit_time.pkl", "rb") as f:
    visited_time = pickle.load(f)

def round_3x4(matrix):
    # First row (index 0, 1, 2, 3)
    matrix[0][0] = round(matrix[0][0] / 1) * 1  # term1
    matrix[0][2] = round(matrix[0][2] / 1) * 1  # term3
    matrix[0][1] = round(matrix[0][1] / 0.04) * 0.04  # term2 0 ~ 0.12 // 0.04 || 4
    matrix[0][3] = round(matrix[0][3] / 0.05) * 0.05  # term4 -0.05 ~ 0.05 // 0.05 || 3
    # Second row (index 1, 2, 3, 4)
    matrix[1][0] = round(matrix[1][0] / 0.01) * 0.01  # term5 -0.16 ~ 0.16 // 0.04 || 9
    matrix[1][1] = round(matrix[1][1] / 0.04) * 0.04  # term6 -0.08 ~ 0.08 // 0.04 || 5
    matrix[1][2] = round(matrix[1][2] / 0.03) * 0.03  # term7 -0.20 ~ 0.20 // 0.05 || 9
    matrix[1][3] = round(matrix[1][3] / 0.03) * 0.03  # term8 -0.04 ~ 0.04 // 0.04 || 3

    matrix[2][0] = round(matrix[2][0] / 0.01) * 0.01  # term5 -0.16 ~ 0.16 // 0.04 || 9
    matrix[2][1] = round(matrix[2][1] / 0.04) * 0.04  # term6 -0.08 ~ 0.08 // 0.04 || 5
    matrix[2][2] = round(matrix[2][2] / 1) * 1  # term7 -0.20 ~ 0.20 // 0.05 || 9
    matrix[2][3] = round(matrix[2][3] / 1) * 1  # term8 -0.04 ~ 0.04 // 0.04 || 3
    return matrix

def find_nearest_kdtree(target_matrix, kdtree, matrix_list):
    target_flat = target_matrix.flatten()
    dist, index = kdtree.query(target_flat)
    return index, matrix_list[index]

flattened_states = [matrix.flatten() for matrix in states]
kdtree = KDTree(flattened_states)


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
        "vehicles_count": 3
        # "initial_lane_id": 0
    },
        "order": "sorted",
        "lanes_count": 2,
        "vehicles_count": 2,
        # "initial_lane_id": 1,
        "absolute": False
}
env = gym.make('highway-fast-v0', render_mode="human", config=config)

for q in range (10):
    test_action_length = 0
    reward1 = 0
    state, _ = env.reset()
    done = False
    Lane = 0
    while not done:
        print("---------------------------------------")
        s1 = round_3x4(state)
        print("state:", s1)
        c1, c2 = find_nearest_kdtree(s1, kdtree, states)
        action = state_policy[c1]
        if visited_time[c1] < 2:
            # print("|new|")
            action = 3
        n_state, reward, done, additional_info, info = env.step(action)
        test_action_length += 1
        print(action, reward, c2)
        a = n_state
        reward1 += reward
        time.sleep(0.1)
        state = n_state
        if additional_info:
            done = True
        print("||||||||||||||||||||||||||||||||||||||||||")
    print(test_action_length,reward1)

from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3 import TD3
import gymnasium

# Make the highway environment
env = gymnasium.make(
        "highway-fast-v0",
        render_mode="human",
        config={
            "action": {
                "type": "ContinuousAction"
            },
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": True
            },
        }
    )

# Load the trained agent
model = TD3.load("TD3_model_1000000.zip", env=env)

# Evaluate the model n number of times
n_eval_episodes = 100
reward, duration = evaluate_policy(model, env,
                                          n_eval_episodes=n_eval_episodes,
                                          return_episode_rewards=True,
                                          deterministic=True)


# Tabulate the results for easy viewing
from tabulate import tabulate

# Define array for number of tests ran when evaluating model
Test_Number = np.arange(1, n_eval_episodes+1)

# Print the table using tabulate
print("\n", tabulate({"Test Number": Test_Number,
                "Reward": reward,
                "Duration (s)": duration}, headers="keys"))

# Print the average reward
print("\nAverage Reward: ", np.mean(reward),"\n")

# Print success rate (duration agent lasted / max duration length)
env_max_duration = 30 # highway-fast_v0 duration is set to 30s
print("Success Rate: ", np.mean(duration)/env_max_duration*100,"% \n")
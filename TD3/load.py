from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3 import TD3
import gymnasium
from tqdm import trange

# Make the highway environment
env = gymnasium.make(
        "highway-fast-v0",
        render_mode="human",
        config={
            "action": {
                "type": "ContinuousAction"
            },
            #"observation": {
            #    "type": "Kinematics",
            #    "features": ["x", "y", "vx", "vy"],
            #    "absolute": True,
            #    "normalize": True
            #},
        }
    )

# Load the trained agent
model = TD3.load("TD3_config_0_new.zip", env=env)

# Evaluate the model n number of times
n_eval_episodes = 5

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



# Record 3 episodes of the trained agent
import base64
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()

def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(
        env, video_folder=video_folder, episode_trigger=lambda e: True
    )

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


env = gymnasium.make(
        "highway-fast-v0",
        render_mode="rgb_array",
        config={
            "action": {
                "type": "ContinuousAction"
            },
            #"observation": {
            #    "type": "Kinematics",
            #    "features": ["x", "y", "vx", "vy"],
            #    "absolute": True,
            #    "normalize": True
            #},
        }
    )
env = record_videos(env)

for episode in trange(3, desc='Test episodes'):
    (obs, info), done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
env.close()
show_videos()

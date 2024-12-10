import gymnasium
import highway_env
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv  # Import SubprocVecEnv
import os
import torch
import time
import multiprocessing
from stable_baselines3.common.evaluation import evaluate_policy

# Create directories
log_dir = "logs"
models_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
"""
config_1
lanes_count: 2
vehicles_count: 50
reward_speed_range: [20, 30]
config_2
4
50
[20, 30]
config_3
6
50
[20, 30]
config_4
2
100
[20, 30]
config_5
4
100
[20, 30]
config_6
6
100
[20, 30]
config_7
2
25
[40, 50]
config_8
4
25
[40, 50]
config_9
6
25
[40, 50]

"""
config_num=0

def make_env():
    """Initialize and configure the environment"""
    env = gymnasium.make(
        "highway-fast-v0",
        render_mode="rgb_array",
        config={
            "action": {
                "type": "ContinuousAction"
            },
            #"lanes_count": 3,
            #"vehicles_count": 25,
            #"reward_speed_range": [40, 50],
        }
    )
    return env

class CustomTD3:
    def __init__(self):
        # Create and wrap the environment
        self.env = DummyVecEnv([make_env])
        #self.env = SubprocVecEnv([make_env() for _ in range(12)])  # 4 parallel environments by default
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        n_actions = self.env.action_space.shape[-1]
        
        # Define action noise for exploration
        self.action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        # Initialize TD3 model
        self.model = TD3(
            "MlpPolicy",
            self.env,
            action_noise=self.action_noise,
            verbose=1,
            buffer_size=100000, # 100000, 500000, 1000000, 5000000, 10000000, 50000, 10000, 1000
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            tensorboard_log=log_dir,
            device=self.device,
            policy_kwargs=dict(net_arch=[256, 256, 128]) # smaller net
        )
    
    def train(self, total_timesteps):
        """Train the model"""
        print("Starting training...")
        try:
            self.model.learn(total_timesteps=total_timesteps, log_interval=1)
            self.save_policy()
            self.model.save(f'TD3_config_{config_num}_new')
        except Exception as e:
            print(f"Training error: {e}")
            self.save_policy()
    
    def save_policy(self):
        """Save the policy network"""
        try:
            policy_path = os.path.join(models_dir, f"td3_policy_config{config_num}_new.pth")
            torch.save(self.model.policy.state_dict(), policy_path)
            print(f"Policy saved to {policy_path}")
        except Exception as e:
            print(f"Error saving policy: {e}")

    def evaluate(self, num_episodes=5):
        """Evaluate the trained policy"""
        print("\nStarting evaluation...")
        total_rewards = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            steps = 0
            done = False

            # Reset environment
            obs = self.env.reset()
            
            while not done:
                # Get action from the policy
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step in the environment
                obs, rewards, dones, infos = self.env.step(action)
                
                # Update episode tracking
                episode_reward += rewards[0]  # rewards is an array for vectorized env
                done = dones[0]  # dones is an array for vectorized env
                steps += 1
                
                # Render if possible
                try:
                    self.env.render()
                except Exception as e:
                    pass  # Silently ignore rendering errors
            
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"\nEvaluation over {num_episodes} episodes:")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Min reward: {min(total_rewards):.2f}")
        print(f"Max reward: {max(total_rewards):.2f}")
        return mean_reward

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'env'):
            try:
                self.env.close()
            except Exception as e:
                print(f"Error closing environment: {e}")


def main():
    agent = None
    try:
        # Initialize agent
        agent = CustomTD3()
        print("Action space:", agent.env.action_space)
        print("Number of actions:", agent.env.action_space.shape[-1])
        
        agent.train(total_timesteps=10_000)
        agent.evaluate(num_episodes=10)

        # Evaluate the model n number of times
        n_eval_episodes = 10
        agent.env.reset()
        reward, duration = evaluate_policy(agent.model, agent.env,
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

        print(f'CONFIG {config_num}')
        
    except Exception as e:
        print(f"Main execution error: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
    finally:
        # Clean up
        if agent is not None:
            agent.close()

if __name__ == "__main__":
    main()

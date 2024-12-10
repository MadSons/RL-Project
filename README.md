# Q-Learning, DQN, and TD3 Implementation for Highway Environments

This repository contains code for implementing Q-Learning, DQN, and TD3 reinforcement learning algorithms in various highway environments. Each folder includes files and instructions for training, testing, visualizing, and saving results. Below is an overview of the folder structure and their respective contents.

---

## Q-Learning
There are 3 folders in the Q learning folder
### Folder: `2_lane_small`
- **`1vmod.ipynb`**  
  - **Part 1 & 2:** Training setup.  
  - **Part 3:** Training process.  
  - **Part 4:** Graphs showing improvement during training.  
  - **Part 5:** Runs for 10,000 episodes to calculate the collision-free rate (adjustable).  
  - **Part 6:** Saves the trained data.
  - **Note:** You can also run only **Part 1, 2, and 5** and adjust the number of episodes in **Part 5** for calculating the collision-free rate over fewer episodes.

- **`1vmod.py`**  
  - Visualization script for human understanding.  
  - Runs the trained model for 10 episodes in a 2-lane, 2-vehicle environment.

- **`.pkl files`**  
  - Saved training data.

---

### Folder: `2_lane_large`
- **`train.ipynb`**  
  - **Part 1, 2, 3:** Training setup.  
  - **Part 4:** Training process.  
  - **Part 5:** Graphs showing improvement during training.  
  - **Part 6:** Saves the trained data.  
  - **Part 7:** Runs for 100 episodes to calculate the collision-free rate (adjustable).

- **`test.ipynb`**  
  - **Part 1:** Collects trained data.  
  - **Part 2:** Tests in a 2-lane, 25-vehicle environment for 10,000 episodes.  
  - **Part 3:** Tests in a 2-lane, 50-vehicle environment for 10,000 episodes.  
  - **Part 4:** Tests in a 2-lane, 100-vehicle environment for 10,000 episodes.

- **`test.py`**  
  - Visualization script for human understanding.  
  - Runs the trained model for 10 episodes in a 2-lane, 50-vehicle environment (takes ~10 seconds to load trained data).

- **`.pkl files`**  
  - Saved training data.

---

### Folder: `4_lane_large`
- **`train.ipynb`**  
  - **Part 1, 2, 3:** Training setup.  
  - **Part 4:** Training process.  
  - **Part 5:** Graphs showing improvement during training.  
  - **Part 6:** Saves the trained data.  
  - **Part 7:** Runs for 100 episodes to calculate the collision-free rate (adjustable).

- **`test.ipynb`**  
  - **Part 1:** Collects trained data.  
  - **Part 2:** Tests in a 4-lane, 50-vehicle environment for 100 episodes.  
  - **Part 3:** Tests in a 4-lane, 25-vehicle environment for 10,000 episodes.  
  - **Part 4:** Tests in a 4-lane, 50-vehicle environment for 10,000 episodes.  
  - **Part 5:** Tests in a 4-lane, 100-vehicle environment for 10,000 episodes.

- **`test.py`**  
  - Visualization script for human understanding.  
  - Runs the trained model for 10 episodes in a 4-lane, 30-vehicle environment (takes ~60 seconds to load trained data).

- **`.pkl files`**  
  - Saved training data.

---

## DQN
There are 6 sections in the .ipynb file. Sections 1-4 should be run in order. Then Section 6 can be run to record episodes of the model under base conditions. Section 5 loops through all of the environment modifications
### Notebook
- **Sections in `.ipynb` file:**  
  1. **Install Environment, Agent, and Libraries:** Installs all dependencies.  
  2. **Train the DQN Model:**  
     - Sets up the environment and trains the DQN model.  
     - Defines the DQN model's parameters and runs the model for a specified number of timesteps.
  3. **Evaluate the Trained Model:**  
     - Evaluates the policy 10 times.  
     - Computes metrics like total reward and success rate (percentage of episodes where collisions were avoided).  
  4. **Run Trained Model for N Episodes, Record Performance Metrics:**  
     - Extra section to explore more performance metrics not included in the report.  
  5. **Modification of Environment Parameters:**  
     - Repeats the training and evaluation process for various environment configurations.
  6. **Record Episode:**  
     - Records three episodes for visualization. Can be run after Section 2 or 5 with a trained model.  
     - These episodes are saved for use in presentations.

---

## TD3
There are 3 python files involving training and viewing results for the TD3 model, td3.py, load.py, and plot_data.py. 

There are also two folders, logs, which contain the tensorboard log files used to plot later, and videos which store some saved videos.

A `requirements.txt` file is included for required Python packages. Install dependencies using:
`pip install -r requirements.txt`
### Files
- **`td3.py`**  
  - Trains and evaluates TD3 models.  
  - Allows for training multiple configurations by manually adjusting configuration variables.  
  - Saves loss values to TensorBoard.  
  - Evaluates the model at the end of training and saves it as a compressed `.zip` file for later use.

- **`load.py`**  
  - Loads saved `.zip` models for additional evaluation.  
  - Attempts to save a video of a TD3 model run on the highway environment.

- **`plot_data.py`**  
  - Extracts TensorBoard data for a manually chosen run and plots results using Matplotlib.

- **`.zip`**  
  - Saved model configurations

### Folders
- **`logs/`**  
  - Contains TensorBoard log files used for plotting.

- **`videos/`**  
  - Stores saved videos of TD3 runs.


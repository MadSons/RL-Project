
TAG_NAME = "train/critic_loss"

import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Define your event file path
log_dir = "logs/TD3_24/"
event_file = next((os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.startswith("events.out")), None)

if event_file is None:
    print("Event file not found.")
else:
    # Initialize variables to store the data
    steps = []
    values = []
    
    # Load the event file
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.tag == TAG_NAME:
                steps.append(event.step)
                values.append(value.simple_value)

    # Plot the data
    plt.plot(steps, values)
    plt.xlabel("Iterations")
    plt.ylabel(TAG_NAME)
    plt.title("TD3 Critic Loss")
    plt.grid()
    plt.savefig('TD3_24_Critic_loss')
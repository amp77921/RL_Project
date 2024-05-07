import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the Q values data from the first file
with open('/Users/austinperales/Downloads/q_values_dqn_ra.pkl', 'rb') as f:
    q_values_log_dqn_ra = pickle.load(f)

# Load the Q values data from the second file
with open('/Users/austinperales/Downloads/q_values_dqn_ma.pkl', 'rb') as f:
    q_values_log_dqn_ma = pickle.load(f)

# Load the Q values data from the third file
with open('/Users/austinperales/Downloads/q_values_mappo_ra.pkl', 'rb') as f:
    q_values_log_mappo_ra = pickle.load(f)

# Load the Q values data from the fourth file (newly added)
with open('/Users/austinperales/Downloads/q_values_mappo_ma.pkl', 'rb') as f:
    q_values_log_mappo_ma = pickle.load(f)

episodes_ra = list(range(1, len(q_values_log_dqn_ra) + 1))
episodes_ma = list(range(1, len(q_values_log_dqn_ma) + 1))
episodes_mappo_ra = list(range(1, len(q_values_log_mappo_ra) + 1))
episodes_mappo_ma = list(range(1, len(q_values_log_mappo_ma) + 1))

# Define the window size for the moving average
window_size = 20

# Calculate moving average for the first set of Q values
moving_avg_ra = np.convolve(q_values_log_dqn_ra, np.ones(window_size)/window_size, mode='valid')

# Calculate moving average for the second set of Q values
moving_avg_ma = np.convolve(q_values_log_dqn_ma, np.ones(window_size)/window_size, mode='valid')

# Calculate moving average for the third set of Q values
moving_avg_mappo_ra = np.convolve(q_values_log_mappo_ra, np.ones(window_size)/window_size, mode='valid')

# Calculate moving average for the fourth set of Q values (newly added)
moving_avg_mappo_ma = np.convolve(q_values_log_mappo_ma, np.ones(window_size)/window_size, mode='valid')

# Create a plot of episodes vs moving averages for all models
plt.figure(figsize=(10, 5))
plt.plot(episodes_ra[window_size-1:], moving_avg_ra, label='DQN RA', color='c')
plt.plot(episodes_ma[window_size-1:], moving_avg_ma, label='DQN MA', color='m')
plt.plot(episodes_mappo_ra[window_size-1:], moving_avg_mappo_ra, label='MA-PPO RA', color='b')
plt.plot(episodes_mappo_ma[window_size-1:], moving_avg_mappo_ma, label='MA-PPO MA', color='pink')  # Newly added plot

plt.ylim(0.4, 1.5) 

plt.title('Moving Averages of Policy Values Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Moving Average Policy Value')
plt.legend()
plt.grid(True)
plt.show()

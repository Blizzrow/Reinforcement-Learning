# Implement generate_session_mountain_car(), training loop, etc.
import gym
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0").env
env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]


agent = MLPClassifier(
    hidden_layer_sizes = (20,20),
    activation = "tanh",
    max_iter=1
)

agent.partial_fit([env.reset()] * n_actions, range(n_actions), range(n_actions))


def generate_session_mountain_car(env, agent, t_max=10000):
    states, actions = [],[]
    total_reward = 0

    s = env.reset()

    for t in range(t_max):
      probs = agent.predict_proba([s])[0]
      a = np.random.choice(np.size(probs), 1, p = probs)[0]
      new_s, r, done, info = env.step(a)
      states.append(s)
      actions.append(a)
      total_reward += r
      s = new_s
      if done:
        break

    return states, actions, total_reward
from IPython.display import clear_output

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):

    elite_states = []
    elite_actions = []

    reward_threshold = np.percentile(rewards_batch, percentile)

    for i in range(0, np.size(rewards_batch)):
      if(rewards_batch[i] >= reward_threshold):
        elite_states.extend(states_batch[i])
        elite_actions.extend(actions_batch[i])

    return elite_states, elite_actions

n_sessions = 100
percentile = 80
log = []

for i in range(0,150):
  sessions = [generate_session_mountain_car(env,agent) for session in range(n_sessions)]
  states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))
 # print("States_Batch", states_batch)
  elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
  #print("Elite States", elite_states[0:5])
  #print("Elite Actions", elite_actions[0:5])
  agent.partial_fit(elite_states, elite_actions)
  print(np.mean(rewards_batch))
  if(np.mean(rewards_batch) > -500):
    print(i)
    env.render()
    print("Won?")

'''
def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10]):
    """
    A convenience function that displays training progress.
    No cool math here, just charts.
    """

    mean_reward = np.mean(rewards_batch)
    threshold = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])

    clear_output(True)
    print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(rewards_batch, range=reward_range)
    plt.vlines([np.percentile(rewards_batch, percentile)],
               [0], [100], label="percentile", color='red')
    plt.legend()
    plt.grid()

    plt.show()

'''

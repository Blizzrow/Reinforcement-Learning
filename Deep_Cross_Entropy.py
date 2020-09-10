import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0").env
env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

from sklearn.neural_network import MLPClassifier

agent = MLPClassifier(
    hidden_layer_sizes=(20, 20),
    activation='tanh',
)

agent.partial_fit([env.reset()] * n_actions, range(n_actions), range(n_actions))

def generate_session(env, agent, t_max=1000):
    """
    Play a single game using agent neural network.
    Terminate when game finishes or after :t_max: steps
    """
    states, actions = [], []
    total_reward = 0

    s = env.reset()

    for t in range(t_max):

      # verwende [s] als argument, weil die Trainingsdaten scheinbar auch [[x1,x2,x3]] als Format hatten
        probs = agent.predict_proba([s])
      #probs[0] wird genutzt, weil wir ein 1D array der warscheinlichkeiten wollen und kein [[p1,p2,p3]]
        probs = probs[0]

        assert probs.shape == (n_actions,), "make sure probabilities are a vector (hint: np.reshape)"

        # use the probabilities you predicted to pick an action
        # sample proportionally to the probabilities, don't just take the most likely action
      #hier wieder [0], weil wir als Output GAR KEIN array haben wollen, sondern einen skalar
        a = np.random.choice(2,1, p = probs)[0]
        # ^-- hint: try np.random.choice

        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward

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
percentile = 70
log = []

for i in range(10):
    # generate new sessions
    sessions = [generate_session(env,agent) for session in range(n_sessions)]

    states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))

    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)

    agent.partial_fit(elite_states, elite_actions)

    #<YOUR CODE: partial_fit agent to predict elite_actions(y) from elite_states(X)>

#    show_progress(rewards_batch, log, percentile, reward_range=[0, np.max(rewards_batch)])

    if np.mean(rewards_batch) > 190:
        print("You Win! You may stop training now via KeyboardInterrupt.")

import gym.wrappers

with gym.wrappers.Monitor(gym.make("CartPole-v0"), directory="videos", force=True) as env_monitor:
    sessions = [generate_session(env_monitor, agent) for _ in range(100)]

from pathlib import Path
from IPython.display import HTML

video_names = sorted([s for s in Path('videos').iterdir() if s.suffix == '.mp4'])

HTML("""
<video width="640" height="480" controls>
  <source src="{}" type="video/mp4">
</video>
""".format(video_names[-1]))  # You can also try other indices

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.q_functions import DQN
from src.displayers import EventDisplayer


def running_avg(x, span: int):
    n_avgs = len(x) - span
    return [np.sum(x[i: i + span]) / span for i in range(n_avgs)]


env = gym.make("MountainCar-v0")
env.seed(1)
torch.manual_seed(1)
np.random.seed(1)


q_function = DQN(
    os_size=env.observation_space.shape[0],
    actions_number=env.action_space.n,
    learning_rate=0.001,
    discount=0.999,
    epsilon=0.3,
    memory_capacity=200,
    memory_sample_size=100
)
displayer = EventDisplayer(20, 50, True)

track_reward = []
track_position = []
n_episodes = 3_000
update_target_every_n = 10
max_position = -0.4


for episode in range(n_episodes):
    done = False
    episode_reward = 0
    episode_loss = 0
    max_pos = env.observation_space.low[0]

    state = env.reset()  # Reset env. to initial state at the beginning of every episode.

    while not done:
        displayer.display(env, episode, starting_from=2000)

        action = q_function.get_action(state)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        max_pos = new_state[0] if new_state[0] > max_pos else max_pos
        goal_reached = new_state[0] >= env.goal_position

        q_function.push_to_memory(state, action, new_state, reward, goal_reached)
        episode_loss += q_function.optimization_step()

        if done:
            track_position.append(max_pos)

            if goal_reached:
                displayer.print_success(episode)
                q_function.reduce_exploration_chance(decay=0.99)
                q_function.reduce_lr()
        else:
            state = new_state

    if episode % update_target_every_n == 0:
        q_function.update_target_policy()
    track_reward.append(episode_reward)
    displayer.print_episode_stats(episode, episode_loss, track_position)

env.close()

plt.plot(running_avg(track_reward, 50))
plt.show()

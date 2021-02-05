'''
Environment state has 2 dimension:
0 is probably the coordinate along the 1D line
1 is probably momentum
'''
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

OS_HIGH = env.observation_space.high
OS_LOW = env.observation_space.low
DISCRETE_OS_SIZE = [20, 20]
DISCRETE_OS_WINDOW = (OS_HIGH - OS_LOW) / DISCRETE_OS_SIZE
LR = 0.2
DISCOUNT = 0.95
EPISODES = 4_000
DISPLAY_EVERY_N = 500
REDUCE_EXPLORATION_EVERY_N = 100
goodness = 0
g_count = 0
EPSILON = 0.3


def running_avg(x, span: int):
    n_avgs = len(x) - span
    return [np.sum(x[i: i + span]) / span for i in range(n_avgs)]


def get_discrete_state(state):
    discrete = (state - OS_LOW) // DISCRETE_OS_WINDOW
    return tuple(discrete.astype(np.int))


def get_next_qvlaue(disc_state):
    if np.random.uniform(0, 1) < EPSILON:
        rnd_action = np.random.randint(0, env.action_space.n)
        return q_table[disc_state + (rnd_action, )]
    else:
        return np.max(q_table[disc_state])


q_table = np.random.uniform(low=-0.5, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
track_reward = []

for episode in range(EPISODES):
    episode_reward = 0
    display = False
    if episode % DISPLAY_EVERY_N == 0:
        print(episode)
        display = True
    done = False
    disc_state = get_discrete_state(env.reset())
    while not done:
        action = np.argmax(q_table[disc_state])
        new_state, reward, done, _ = env.step(action)
        new_disc_state = get_discrete_state(new_state)

        if display:
            env.render()
        episode_reward += reward

        if not done:
            # Compute Q-learning update.
            next_q = get_next_qvlaue(new_disc_state)
            current_q = q_table[disc_state + (action, )]
            new_q = (1 - LR) * current_q + LR * (reward + DISCOUNT * next_q)
            # Update Q for this state and action.
            q_table[disc_state + (action, )] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[disc_state + (action, )] = 0

            # Measure how fast the agent learns.
            if g_count < 10:
                goodness += episode
                g_count += 1

        disc_state = new_disc_state

    if EPSILON >= 0 and EPSILON % REDUCE_EXPLORATION_EVERY_N == 0:
        EPSILON -= 0.01

    track_reward.append(episode_reward)

print()
print(goodness / g_count)

env.close()

plt.plot(running_avg(track_reward, 50))
plt.show()

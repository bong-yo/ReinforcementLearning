import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange
import numpy as np


env = gym.make('MountainCar-v0')
env.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 200
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)
        self.f = nn.GELU()

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            self.f,
            self.l2,
        )
        return model(x)


# Parameters
steps = 2000
state = env.reset()
epsilon = 0.3
gamma = 0.99
loss_history = []
reward_history = []
episodes = 3000
max_position = -0.4
learning_rate = 0.001
successes = 0
position = []

# Initialize Policy
policy = Policy()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

for episode in trange(episodes):
    episode_loss = 0
    episode_reward = 0
    state = env.reset()

    for s in range(steps):
        # # Uncomment to render environment
        # if episode % 100 == 0 and episode > 0:
        #     env.render()

        # Get first action value function
        Q = policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))

        # Choose epsilon-greedy action
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, 3)
        else:
            _, action = torch.max(Q, -1)
            action = action.item()

        # Step forward and receive next state and reward
        state_1, reward, done, _ = env.step(action)

        # Find max Q for t+1 state
        Q1 = policy(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
        maxQ1, _ = torch.max(Q1, -1)

        # Create target Q value for training the policy
        Q_target = Q.clone()
        Q_target = Variable(Q_target.data)
        Q_target[action] = reward + torch.mul(maxQ1.detach(), gamma)

        # Calculate loss
        loss = loss_fn(Q, Q_target)

        # Update policy
        policy.zero_grad()
        loss.backward()
        optimizer.step()

        # Record history
        episode_loss += loss.item()
        episode_reward += reward
        # Keep track of max position
        if state_1[0] > max_position:
            max_position = state_1[0]

        if done:
            if state_1[0] >= 0.5:
                # On successful epsisodes, adjust the following parameters

                # Adjust epsilon
                epsilon *= .99
                # Adjust learning rate
                scheduler.step()
                # Record successful episode
                successes += 1
                print(f"\nSuccess at ep: {episode}")

            # Record history
            loss_history.append(episode_loss)
            reward_history.append(episode_reward)
            position.append(state_1[0])

            break
        else:
            state = state_1


import matplotlib.pyplot as plt
import pandas as pd
plt.figure(2, figsize=[10, 5])
p = pd.Series(position)
ma = p.rolling(10).mean()
plt.plot(p, alpha=0.8)
plt.plot(ma)
plt.xlabel('Episode')
plt.ylabel('Position')
plt.title('Car Final Position')
plt.savefig('Final Position.png')
plt.show()

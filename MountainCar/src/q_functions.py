import numpy as np
import torch
from torch import FloatTensor
from torch.nn import SmoothL1Loss, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from typing import Tuple
from src.state_handler import StateHandler
from src.networks import Policy
from src.utils import create_batches, unzip_to_tensors
from src.memory_replay import MemoryReplay

torch.manual_seed(1)
np.random.seed(1)


class QTable(StateHandler):
    def __init__(self, os_low: int, os_high: int, dicrete_os_size: Tuple[int, int],
                 action_space_n: int, discount: float, learning_rate: float = 0.1,
                 epsilon: float = 0.3, delta_epsilon: float = 0.01,
                 reduce_epsilon_every_n: int = 100):
        super(QTable, self).__init__(os_low, os_high, dicrete_os_size)
        self.epsilon = epsilon  # Exploration chance parameter.
        self.gamma = discount
        self.lr = learning_rate
        self.delta_epsilon = delta_epsilon
        self.reduce_epsilon_every_n = reduce_epsilon_every_n
        self.action_space_n = action_space_n
        self.table = np.zeros(shape=(dicrete_os_size + (action_space_n, )))
        self.inputs = []
        self.targets = []

    def get_action(self, state):
        '''Allow for randomness (epsilon) in the choice of action.'''
        state_disc = self.get_discrete_state(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_space_n)
        else:
            return np.argmax(self.table[state_disc])

    def get_qvalue(self, state, action):
        state_disc = self.get_discrete_state(state)
        return self.table[state_disc + (action, )]

    def get_max_qvalue(self, state):
        state_disc = self.get_discrete_state(state)
        return np.max(self.table[state_disc])

    def save_input_target(self, state, action, next_state, reward):
        self.inputs.append((self.get_discrete_state(state), action))
        current_q = self.get_qvalue(state, action)  # Since we allow for random action, Q-value is not always = max(Q-value) for the state.
        next_q = self.get_max_qvalue(next_state) if next_state is not None else 0
        new_q_value = \
            (1 - self.lr) * current_q + self.lr * (reward + self.gamma * next_q)
        self.targets.append(new_q_value)

    def learn_from_episode(self):
        for (state_disc, action), target in zip(self.inputs, self.targets):
            self.table[state_disc + (action, )] = target
        # Reset inputs and targets.
        self.inputs = []
        self.targets = []

    def reduce_exploration_chance(self, episode_num):
        if self.epsilon > 0 and episode_num % self.reduce_epsilon_every_n == 0:
            self.epsilon -= self.delta_epsilon


class QNN:
    def __init__(self, os_size: int, actions_number: int, learning_rate: float,
                 discount: float, epsilon: float = 0.3):
        self.n_actions = actions_number
        self.epsilon = epsilon
        self.gamma = discount

        self.policy = Policy(os_size, 200, self.n_actions)
        self.criterion = MSELoss()
        self.optimizer = SGD(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.data = []

    def get_action(self, Q):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                _, action = torch.max(Q, -1)
                return action.item()

    def get_qvalue(self, state):
        return self.policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))

    def get_max_qvalue(self, state):
        return torch.max(self.get_qvalue(state), -1)[0]

    def save_input_target(self, Q, action, next_state, reward):
        self.data.append((Q, action, next_state, reward))

    def reduce_exploration_chance(self, decay):
        if self.epsilon > 0:
            self.epsilon *= decay

    def reduce_lr(self):
        self.scheduler.step()

    def learn_form_step(self, Q, action, next_state, reward):
        with torch.no_grad():
            Q_target = Q.clone()
            Q_target = Variable(Q_target.data)
            Q_target[action] = reward + \
                torch.mul(self.get_max_qvalue(next_state).detach(), self.gamma)

        loss = self.criterion(Q, Q_target)
        self.policy.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def learn_from_episode(self):
        loss_sum = 0
        for Q_b, action_b, next_state_b, reward_b in create_batches(self.data, 300):

            Q_batch = torch.vstack(Q_b)

            with torch.no_grad():
                next_state_b = np.vstack(next_state_b)
                reward_b = torch.FloatTensor(reward_b)
                Q_target = Q_batch.clone()
                Q_target = Variable(Q_target.data)
                Q_target[range(len(action_b)), action_b] = reward_b + \
                    torch.mul(self.get_max_qvalue(next_state_b).detach(), self.gamma)

            loss = self.criterion(Q_batch, Q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()

        # Reset inputs and targets.
        self.data = []
        return loss_sum


class DQN(MemoryReplay):
    def __init__(self, os_size: int, actions_number: int, learning_rate: float,
                 discount: float, epsilon: float, memory_capacity: int,
                 memory_sample_size: int):
        super(DQN, self).__init__(memory_capacity)
        self.mem_sample_size = memory_sample_size
        self.n_actions = actions_number
        self.epsilon = epsilon
        self.gamma = discount

        self.policy = Policy(os_size, 200, self.n_actions)
        with torch.no_grad():
            self.target_policy = Policy(os_size, 200, self.n_actions)
            self.update_target_policy()
            self.target_policy.eval()
        self.criterion = MSELoss()
        self.optimizer = SGD(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                Q = self.get_qvalue(state)
                _, action = torch.max(Q, -1)
                return action.item()

    def get_qvalue(self, state):
        return self.policy(Variable(FloatTensor(state)))

    def get_max_qvalue(self, state):
        return torch.max(self.get_qvalue(state), -1)[0]

    def save_input_target(self, Q, action, next_state, reward):
        self.data.append((Q, action, next_state, reward))

    def reduce_exploration_chance(self, decay):
        if self.epsilon > 0:
            self.epsilon *= decay

    def reduce_lr(self):
        self.scheduler.step()

    def update_target_policy(self):
        with torch.no_grad():
            self.target_policy.load_state_dict(self.policy.state_dict())
            self.target_policy.eval()

    def optimization_step(self):
        if len(self.memory) < self.mem_sample_size:
            return 0

        batch = self.sample_memory(size=self.mem_sample_size)
        state_b, action_b, next_state_b, reward_b, goal_reached = \
            unzip_to_tensors(batch)

        Q_batch = self.get_qvalue(np.array(state_b))
        Q_batch = Q_batch[range(len(action_b)), action_b]
        with torch.no_grad():
            Qnext_batch = self.target_policy(Variable(next_state_b))
            max_future_reward = torch.max(Qnext_batch, -1)[0]
            target_batch = reward_b + \
                self.gamma * max_future_reward.detach() * (1 - goal_reached)  # Set to 0 the max_future_reward of states where the goal has been reached.

        loss = self.criterion(Q_batch, target_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

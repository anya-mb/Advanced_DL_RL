import random
import numpy as np
from collections import namedtuple, deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import matplotlib.pyplot as plt

TAU = 0.9
BATCH_SIZE = 256
HIDDEN_SIZE = 128
BUFFER_SIZE = 10000

UPDATE_EVERY = 4
UPDATE_TARGET_EVERY = 1000


# working with env
def get_used_cells(state):
    return [ind for ind, value in enumerate(state) if value != '1']


def get_empty_cells(state):
    return [ind for ind, value in enumerate(state) if value == '1']


# models
class DQNet(nn.Module):
    def __init__(self, game_size, action_size, inner_size=128):
        super(DQNet, self).__init__()
        assert game_size ** 2 == action_size, 'wrong game_size or action_size'

        self.inner_size = inner_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=inner_size, kernel_size=game_size)
        self.fc1 = nn.Linear(inner_size, inner_size)
        self.fc2 = nn.Linear(inner_size, action_size)

    def forward(self, state):
        x = self.conv(state).view(-1, self.inner_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DuelingDQNet(nn.Module):
    def __init__(self, game_size, action_size, inner_size=128):
        super(DuelingDQNet, self).__init__()
        assert game_size ** 2 == action_size, 'wrong game_size or action_size'

        self.inner_size = inner_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=inner_size, kernel_size=game_size)
        self.fc1 = nn.Linear(inner_size, inner_size)
        self.fc_adv = nn.Linear(inner_size, action_size)
        self.fc_val = nn.Linear(inner_size, 1)

    def forward(self, state):
        x = self.conv(state).view(-1, self.inner_size)
        x = self.fc1(x)
        x = F.relu(x)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        result = val + (adv - adv.mean())
        return result


# buffer for better work with transitions (state, action, reward, next_state, done)
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device

    def consume_transition(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        transitions = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([t.state for t in transitions if t is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([t.action for t in transitions if t is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in transitions if t is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in transitions if t is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([t.done for t in transitions if t is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# agents
class RandomAgent:
    # select random available action
    def select_best_action(self, state):
        return random.choice(get_empty_cells(state))


class DQNAgent:
    # selects available actions based on trained DQN model
    def __init__(self, model, game_size, action_size, 
                 lr, gamma, epsilon, side, 
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 update_every=UPDATE_EVERY, 
                 target_update_every=UPDATE_TARGET_EVERY, 
                 inner_size=HIDDEN_SIZE):
        self.game_size = game_size
        self.gamma = gamma
        self.eps = epsilon

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Training on device:", self.device)

        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)

        self.side = side

        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update_every = target_update_every
        
        self.model_local = model(game_size, action_size, inner_size).to(self.device)
        self.model_target = model(game_size, action_size, inner_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model_local.parameters(), lr=lr)

        self.current_action = None
        self.current_state = None
        self.t_step = 0 

        self.model_target.load_state_dict(deepcopy(self.model_local.state_dict()))
        
    # @staticmethod
    def state_to_2d_array(self, state, size):
        # for 2d convolutions to work
        return np.array([float(number) for number in state]).reshape(size, size)
    
    def update_memory(self, next_state, next_action, reward, done):
        # update memory starting with the second action
        if self.current_action:
            self.memory.consume_transition(
                self.state_to_2d_array(self.current_state, self.game_size),
                self.current_action,
                reward, 
                self.state_to_2d_array(next_state, self.game_size),
                done)

        self.current_state = next_state
        self.current_action = next_action

    def step(self, state, action, reward, done):
        # storing data and training based on random sample from stored data
        self.update_memory(state, action, reward, done)
        self.t_step += 1

        if (self.t_step % self.update_every) == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        if (self.t_step % self.target_update_every) == 0:
            self.update_target_network(self.model_local, self.model_target, TAU)
    
    def learn(self, experiences):
        # training DQN model
        states, actions, rewards, next_states, dones = experiences

        states = states.view(-1, 1, self.game_size, self.game_size)
        next_states = next_states.view(-1, 1, self.game_size, self.game_size)

        # # training step, update DQN by experiences batch
        q_expected = self.model_local(states).gather(1, actions)
        q_targets_next = self.model_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  

    def update_target_network(self, local_model, target_model, tau=TAU):
        # trick for stability
        # θ_target = τ*θ_local + (1 - τ)*θ_target 

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)      
    
    def select_best_action(self, state):
        # greedy
        modified_state = self.state_to_2d_array(state, self.game_size)
        state_tensor = torch.from_numpy(modified_state).float().unsqueeze(0).to(self.device)

        self.model_local.eval()
        with torch.no_grad():
            action_values = self.model_target(state_tensor).flatten()
        self.model_local.train()

        action_values[get_used_cells(state)] = -np.inf
        result = np.argmax(action_values.cpu().data.numpy())

        return result
    
    def select_action(self, state):
        # epsilon-greedy
        if np.random.rand() < self.eps:
            return random.choice(get_empty_cells(state))
        return self.select_best_action(state)


class DoubleDQNAgent(DQNAgent):
    def __init__(self, model, game_size, action_size,
                 lr, gamma, epsilon, side,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 update_every=UPDATE_EVERY,
                 target_update_every=UPDATE_TARGET_EVERY,
                 inner_size=HIDDEN_SIZE):

        super().__init__(model, game_size, action_size,
                 lr, gamma, epsilon, side,
                 buffer_size=buffer_size, batch_size=batch_size,
                 update_every=update_every,
                 target_update_every=target_update_every,
                 inner_size=inner_size)

        self.model_A = model(game_size, action_size, inner_size).to(self.device)
        self.model_B = model(game_size, action_size, inner_size).to(self.device)
        self.optimizer_A = torch.optim.Adam(self.model_A.parameters(), lr=lr)
        self.optimizer_B = torch.optim.Adam(self.model_B.parameters(), lr=lr)

    def step(self, state, action, reward, done):
        self.update_memory(state, action, reward, done)
        self.t_step += 1

        if (self.t_step % self.update_every) == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = states.view(-1, 1, self.game_size, self.game_size)
        next_states = next_states.view(-1, 1, self.game_size, self.game_size)

        # # training step, update DQN by experiences batch
        # choose randomly to update model A or B
        if np.random.rand() < 0.5:
            q_expected = self.model_A(states).gather(1, actions)
            q_targets_next = self.model_B(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

            loss = F.mse_loss(q_expected, q_targets)
            self.optimizer_A.zero_grad()
            loss.backward()
            self.optimizer_A.step()

        else:
            q_expected = self.model_B(states).gather(1, actions)
            q_targets_next = self.model_A(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

            loss = F.mse_loss(q_expected, q_targets)
            self.optimizer_B.zero_grad()
            loss.backward()
            self.optimizer_B.step()


# training and evaluation functions
def train(env, x_agent, o_agent, n_episodes):
    # train against itself, x vs o
    agents = {1: x_agent, -1: o_agent}

    for _ in range(n_episodes):
        env.reset()        
        state, empty_cells, turn = env.getState()
        agents[1].current_action = None
        agents[-1].current_action = None
        done = False
        
        while not done:
            action = agents[turn].select_action(state)
            agents[turn].update_memory(state, action, 0, done)
            (state, empty_cells, turn), reward, done, _ = env.step(env.action_from_int(action))
            
        agents[1].step(state, action, reward, done)
        agents[-1].step(state, action, reward, done)


def evaluate(env, dqn_agent, n_episodes):
    # evaluate against random agent
    if dqn_agent.side == 'o':
        agents = {1: RandomAgent(), -1: dqn_agent}
    else:
        agents = {1: dqn_agent, -1: RandomAgent()}
        
    all_rewards = []
    
    for _ in range(n_episodes):
        env.reset()
        state, empty_cells, turn = env.getState()
        done = False

        while not done:
            action = agents[turn].select_best_action(state)
            (state, empty_cells, turn), reward, done, _ = env.step(env.action_from_int(action))
        all_rewards.append(reward)
    
    return all_rewards


def train_dqn_agents(env, x_agent, o_agent, plot_rewards=True,
                     n_epochs=100, n_episodes_train=500,
                     n_episodes_evaluate=50):
  
    x_mean_rewards = []
    o_mean_rewards = []

    for _ in tqdm.tqdm(range(n_epochs)):
        train(env, x_agent, o_agent, n_episodes_train)

        x_mean_rewards.append(np.mean(evaluate(env, x_agent, n_episodes_evaluate)))
        o_mean_rewards.append(np.mean(evaluate(env, o_agent, n_episodes_evaluate)))

    if plot_rewards:
        plt.plot(x_mean_rewards, label='X agent')
        plt.plot(o_mean_rewards, label='O agent')
        plt.title(f'DQN agent vs random player')
        plt.xlabel('Epoch')
        plt.ylabel('Mean reward')
        plt.legend()
        plt.show()

    print(f"X mean reward last 10 epochs: {np.mean(x_mean_rewards[:-10])}")
    print(f"O mean reward last 10 epochs: {np.mean(o_mean_rewards[:-10])}")

    return x_agent, o_agent

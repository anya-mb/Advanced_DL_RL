import random
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict


# function for game demo
def show_game(env, agent):
    env.reset()

    state, empty_cells, turn = env.getState()
    done = False

    while not done:
        action = agent.select_best_action(state)
        (state, empty_cells, turn), reward, done, _ = env.step(env.action_from_int(action))
        env.printBoard()
        print(f"State: {state}, turn: {turn}, reward: {reward}")

    winner = env.getWinner()

    if winner == 1:
        print("Crosses X won")
    elif winner == -1:
        print("Naughts O won")
    else:
        print("Draw")


def get_empty_cells(state):
    return [ind for ind, value in enumerate(state) if value == '1']


def get_used_cells(state):
    return [ind for ind, value in enumerate(state) if value != '1']


class RandomAgent:
    def select_best_action(self, state):
        return random.choice(get_empty_cells(state))


class TabularQAgent:
    # Tabular Q-learning, learning the table of scores
    def __init__(self, number_of_states, lr, gamma, eps, side):
        # init Q-table with zeros by default
        self.Q = defaultdict(lambda: np.zeros(number_of_states))

        self.lr = lr
        self.gamma = gamma
        self.eps = eps

        self.side = side

        self.current_action = None
        self.current_state = None
    
    def select_best_action(self, state):
        # greedy action
        # used cells are not reachable
        self.Q[state][get_used_cells(state)] = -np.inf

        return np.argmax(self.Q[state])
    
    def select_action(self, state):
        # epsilon-greedy action
        if np.random.rand() < self.eps:
            return random.choice(get_empty_cells(state))
        else:
            return self.select_best_action(state)
    
    def update_Q(self, next_state, next_action, reward):
        # update the table of scores
        if self.side == 'o':
            # to make it positive
            reward = -1. * reward

        # update Q-value starting from the second step
        if self.current_action:
            current_Q = self.Q[self.current_state][self.current_action]
            Qsa_optimal_furure_value = np.max(self.Q[next_state])

            self.Q[self.current_state][self.current_action] = current_Q \
              + self.lr * (reward + (self.gamma * Qsa_optimal_furure_value) - current_Q)

        self.current_state = next_state
        self.current_action = next_action


# training and evaluation functions
def train(env, x_agent, o_agent, n_episodes):
    # train tabular Q agent against itself of other side, x vs o
    agents = {1: x_agent, -1: o_agent}

    for _ in range(n_episodes):
        env.reset()
        state, empty_cells, turn = env.getState()
        agents[1].curr_action = None
        agents[-1].curr_action = None
        done = False
        
        # play one episode
        while not done:
            action = agents[turn].select_action(state)
            agents[turn].update_Q(state, action, 0)
            (state, empty_cells, turn), reward, done, _ = env.step(env.action_from_int(action))

        agents[1].update_Q(state, action, reward)
        agents[-1].update_Q(state, action, reward)


def evaluate(env, tq_agent, n_episodes):
    # evaluate against random
    if tq_agent.side == 'o':
        agents = {1: RandomAgent(), -1: tq_agent}
        multiplier = -1. # naughts best reward is -1
    else:
        agents = {1: tq_agent, -1: RandomAgent()}
        multiplier = 1. # crosses best reward is 1

    all_rewards = []
    
    for _ in range(n_episodes):
        env.reset()
        state, empty_cells, turn = env.getState()
        done = False
        while not done:
            action = agents[turn].select_best_action(state)
            (state, empty_cells, turn), reward, done, _ = env.step(env.action_from_int(action))
            reward = multiplier * reward
        all_rewards.append(reward)
    
    return all_rewards


def train_q_agents(env, x_agent, o_agent, plot_rewards=True,
                   n_epochs=300, n_episodes_train=1000,
                   n_episodes_evaluate=100):
  
    x_mean_rewards = []
    o_mean_rewards = []

    for _ in tqdm.tqdm(range(n_epochs)):
        train(env, x_agent, o_agent, n_episodes_train)

        x_mean_rewards.append(np.mean(evaluate(env, x_agent, n_episodes_evaluate)))
        o_mean_rewards.append(np.mean(evaluate(env, o_agent, n_episodes_evaluate)))

    if plot_rewards:
        plt.plot(x_mean_rewards, label='X agent')
        plt.plot(o_mean_rewards, label='O agent')
        plt.title(f'Tabular Q-learning vs random player ')
        plt.xlabel('Epoch')
        plt.ylabel('Mean reward')
        plt.legend()
        plt.show()

    print(f"X mean reward last 10 epochs: {np.mean(x_mean_rewards[:-10])}")
    print(f"O mean reward last 10 epochs: {np.mean(o_mean_rewards[:-10])}")

    return x_agent, o_agent

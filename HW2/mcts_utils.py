import numpy as np
import random
import tqdm
from copy import deepcopy
import math
from collections import defaultdict
import matplotlib.pyplot as plt


# working with env
def get_used_cells(state):
    return [ind for ind, value in enumerate(state) if value != '1']


def get_empty_cells(state):
    return [ind for ind, value in enumerate(state) if value == '1']


# agents
class RandomAgent:
    # select random available action
    def select_best_action(self, state):
        return random.choice(get_empty_cells(state))


class RolloutAgent:
    # build rollouts vs random choices

    def __init__(self, env, n_rollouts, side):
        self.env = env
        self.opp_agent = RandomAgent()
        self.side = side
        self.n_rollouts = n_rollouts

    def make_rollout(self, env):
        roll_env = deepcopy(env)
        state, empty_cells, turn = roll_env.getState()
        roll_side = -turn
        done = False

        while not done:
            action = self.opp_agent.select_best_action(state)
            (state, empty_cells, turn), reward, done, _ = roll_env.step(env.action_from_int(action))

        return reward * roll_side

    def eval_by_rollouts(self):
        state, empty_cells, turn = self.env.getState()
        curr_side = turn
        action_estimates = {}

        for action in get_empty_cells(state):

            current_env = deepcopy(self.env)
            (state, empty_cells, turn), reward, done, _ = current_env.step(current_env.action_from_int(action))

            if done:
                action_estimates[action] = reward * curr_side
            else:
                action_estimates[action] = np.mean([self.make_rollout(current_env) for _ in range(self.n_rollouts)])

        return max(action_estimates, key=action_estimates.get)

    def run_episode(self):
        self.env.reset()
        state, empty_cells, turn = self.env.getState()
        multiplier = 1 if self.side == 'x' else -1
        done = False

        while not done:
            if self.side == 'x':

                action = self.eval_by_rollouts() if turn == 1 \
                    else self.opp_agent.select_best_action(state)

            else:
                action = self.eval_by_rollouts() if turn == -1 \
                    else self.opp_agent.select_best_action(state)
            (state, empty_cells, turn), reward, done, _ = self.env.step(self.env.action_from_int(action))

        return reward * multiplier


class MCTSAgent:
    # Monte Carlo Tree Search
    def __init__(self, env, side='x', exploration_weight=1):
        self.env = env
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.C = exploration_weight
        self.side = 1 if side == 'x' else -1

    def select_action(self, node1, node2):
        node1_arr = np.array(list(node1)).reshape(self.env.n_rows, self.env.n_cols)
        node2_arr = np.array(list(node2)).reshape(self.env.n_rows, self.env.n_cols)

        action_idx = np.argwhere(node1_arr != node2_arr)
        action_idx = action_idx.flatten()

        return action_idx

    def uct_select(self, node):
        # Upper Confidence Bounds estimation
        log_N_vertex = math.log(self.N[node])
        uct = []

        for child in self.children[node]:
            uct.append(self.Q[child] / self.N[child] + \
                       self.C * math.sqrt(log_N_vertex / self.N[child]))

        return self.children[node][np.argmax(uct)]

    def select_node(self, node):
        # Find an unexplored descendant of `node`
        path = []

        while True:
            path.append(node)

            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path

            for child in self.children[node]:
                if child not in self.N:
                    # select first unexplored child
                    action = self.select_action(node, child)
                    path.append(child)
                    self.env.step(action)
                    return path

            if self.side == self.env.curTurn:
                # continue selecting by UCT
                child = self.uct_select(node)
                action = self.select_action(node, child)
            else:

                # opposite agent makes random action
                action = random.choice(self.env.getEmptySpaces())
            (node, _, _), _, _, _ = self.env.step(action)

    def expand(self):
        # Update the `children` dict with the children of `node`
        node, empty_cells, turn = self.env.getState()
        if node in self.children:
            return  # already expanded
        children = []
        if not self.env.gameOver:
            for action in empty_cells:
                self.env.makeMove(turn, action[0], action[1])
                children.append(self.env.getHash())
                self.env.makeMove(0, action[0], action[1])

        self.children[node] = children

    def backprop(self, path, reward):
        # Send the reward back up
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            # inverse for enemy
            reward = 1 - reward

    def make_rollout(self):
        if self.env.gameOver:
            self.env.curTurn = -self.env.curTurn
            reward = self.env.isTerminal()
            return int(reward * self.env.curTurn > 0), True

        env = deepcopy(self.env)
        _, empty_cells, turn = env.getState()
        done = False

        while not done:
            action = random.choice(empty_cells)
            (_, empty_cells, _), reward, done, _ = env.step(action)

        fin_reward = None

        if turn == 1:
            fin_reward = 0 if reward >= 0 else 1
        elif turn == -1:
            fin_reward = 0 if reward <= 0 else 1

        return fin_reward, False

    def select_best_action(self):
        # greedy action
        node, empty_cells, _ = self.env.getState()

        if node not in self.children:
            return random.choice(empty_cells)

        # get mean Q for each available action
        Q_values = []

        for child in self.children[node]:
            if child not in self.N:
                Q_values.append(-np.inf)
            else:
                Q_values.append(self.Q[child] / self.N[child])

        best_action = empty_cells[np.argmax(Q_values)]

        return best_action

    def learn(self, n_episodes):
        # MCTS learning
        for _ in range(n_episodes):
            self.env.reset()
            done = False
            path = []

            while not done:
                # step 1. Selection
                node, _, _ = self.env.getState()
                curr_path = self.select_node(node)
                path += curr_path

                # step 2. Expansion
                self.expand()

                # step 3. Simulation
                reward, done = self.make_rollout()

                # step 4. Backpropagation
                self.backprop(path, reward)
                path = path[:-1]

    def evaluate(self, n_episodes):
        # evaluate against random
        all_rewards = []

        for _ in range(n_episodes):
            self.env.reset()
            state, empty_cells, turn = self.env.getState()
            done = False

            while not done:
                if (turn == 1 and self.side == 1) or (turn == -1 and self.side == -1):
                    action = self.select_best_action()
                else:
                    action = random.choice(empty_cells)

                (state, empty_cells, turn), reward, done, _ = self.env.step(action)

            if reward == -10:
                print(state, empty_cells)
                raise ValueError('Incorrect action')

            fin_reward = reward if self.side == 1 else -reward
            all_rewards.append(fin_reward)

        return all_rewards


# train functions
def train_rollouts_agents(env, x_agent, o_agent, n_iter, plot_rewards=True, n_epochs=300):
    x_mean_rewards = []
    o_mean_rewards = []
    x_means = []
    o_means = []

    for i in tqdm.tqdm(range(n_epochs)):
        x_mean_rewards.append(x_agent.run_episode())
        o_mean_rewards.append(o_agent.run_episode())
        if (i + 1) % 10 == 0:
            x_means.append(np.mean(x_mean_rewards))
            o_means.append(np.mean(o_mean_rewards))

    if plot_rewards:
        plt.plot(x_means, label='X agent')
        plt.plot(o_means, label='O agent')
        plt.title(f'Rollouts n_iter={n_iter}')
        plt.xlabel('Epoch')
        plt.ylabel('Cumulative mean reward')
        plt.legend()
        plt.show()

    print(f"X mean reward last 10 epochs: {np.mean(x_means[:-10])}")
    print(f"O mean reward last 10 epochs: {np.mean(o_means[:-10])}")
    return x_agent, o_agent


def train_mcts_agents(env, x_agent, o_agent, c, plot_rewards=True,
                   n_epochs=300, n_episodes_train=1000,
                   n_episodes_evaluate=100):

    x_mean_rewards = []
    o_mean_rewards = []
    x_means = []
    o_means = []

    for _ in tqdm.tqdm(range(n_epochs)):
        x_agent.learn(n_episodes_train)
        o_agent.learn(n_episodes_train)
        x_mean_rewards.append(x_agent.evaluate(n_episodes_evaluate))
        o_mean_rewards.append(o_agent.evaluate(n_episodes_evaluate))
        x_means.append(np.mean(x_mean_rewards))
        o_means.append(np.mean(o_mean_rewards))

    if plot_rewards:
        plt.plot(x_means, label='X agent')
        plt.plot(o_means, label='O agent')
        plt.title(f'MCTS C={round(c, 2)}')
        plt.xlabel('Epoch')
        plt.ylabel('Cumulative mean reward')
        plt.legend()
        plt.show()

    print(f"X mean reward last 10 epochs: {np.mean(x_means[:-10])}")
    print(f"O mean reward last 10 epochs: {np.mean(o_means[:-10])}")
    return x_agent, o_agent

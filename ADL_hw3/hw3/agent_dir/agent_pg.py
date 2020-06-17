import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []

    def make_action(self, state, test=False):
        # TODO: Replace this line!
        # 1. Use your model to output distribution over actions and sample from it.
        #    HINT: torch.distributions.Categorical 
        # 2. Save action probability in self.saved_action
        state = torch.from_numpy(state).float().unsqueeze(0) # change to tensor dtype
        probs = self.model(state) # feedin model
        move_dist = Categorical(probs) # create probs distribution

        action = move_dist.sample() # sample action from dist.
        self.saved_actions.append(move_dist.log_prob(action)) # save sampled action with log prob
        return action.item()

    def update(self):
        # init
        R = 0
        loss = []
        returns = []
        eps = np.finfo(np.float32).eps.item()
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns) 
        returns = (returns - returns.mean()) / (returns.std() + eps) # normalization
        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        for log_p, R in zip(self.saved_actions, returns):
            loss.append(-log_p * R)
        loss = torch.cat(loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self):
        self.record_reward = []

        avg_reward = None
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            self.rewards = []
            self.saved_actions = []
            self.record_reward.append(avg_reward)

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))

            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                self.record_reward = np.array(self.record_reward)
                np.save('pg_reward', self.record_reward)
                self.save('pg.cpt')
                break

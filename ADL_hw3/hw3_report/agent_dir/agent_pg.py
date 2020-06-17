import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Categorical

REWARD_STOP = 100

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # action_prob = F.softmax(x, dim=1)
        return x

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # action_prob = F.softmax(x, dim=1)
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.actor = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        self.critic = CriticNet(state_dim = self.env.observation_space.shape[0],
                                action_num= self.env.action_space.n,
                                hidden_dim=64)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        print(self.actor)
        print(self.critic)

        if args.test_pg:
            self.load('pg_actor.cpt', 'pg_critic.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-03)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-03)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []

        # PPO h-para
        self.ppo_steps = 5
        self.ppo_clip = 0.2


    def save(self, save_path_actor, save_path_critic):
        print('save model to', save_path_actor)
        print('save model to', save_path_critic)
        torch.save(self.actor.state_dict(), save_path_actor)
        torch.save(self.critic.state_dict(), save_path_critic)


    def load(self, load_path_actor, load_path_critic):
        print('load model from', load_path_actor)
        print('load model from', load_path_critic)
        self.actor.load_state_dict(torch.load(load_path_actor))
        self.critic.load_state_dict(torch.load(load_path_critic))

    def init_game_setting(self):
        self.rewards, self.saved_actions, self.states= [], [], []
        self.actions = []
        self.values = []

    def make_action(self, state, test=False):
        # TODO: Replace this line!
        # 1. Use your model to output distribution over actions and sample from it.
        #    HINT: torch.distributions.Categorical 
        # 2. Save action probability in self.saved_action
        
        state = torch.from_numpy(state).float().unsqueeze(0) # change to tensor dtype
        self.states.append(state)

        actor_pred = self.actor(state) # feedin both model
        critic_pred = self.critic(state)
        actor_prob = F.softmax(actor_pred, dim = -1)
        
        move_dist = Categorical(actor_prob) # create probs distribution

        if test == False:
            action = move_dist.sample() # sample action from dist.
        else:
            action = torch.argmax(actor_prob, dim = -1)

        # action = move_dist.sample()

        self.actions.append(action)
        self.values.append(critic_pred)
        self.saved_actions.append(move_dist.log_prob(action)) # save sampled action with log prob
        
        return action.item()
        

    def update(self):
        # init
        R = 0
        returns = []
        eps = np.finfo(np.float32).eps.item()
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        # get returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns) 
        returns = (returns - returns.mean()) / (returns.std() + eps) # normalization
        # get advantages
        advantages = returns - self.values
        advantages = (advantages - advantages.mean()) / advantages.std()

        # update policy
        total_policy_loss, total_value_loss = 0,0
        
        self.states = self.states.detach()
        self.actions = self.actions.detach()
        self.saved_actions = self.saved_actions.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        for i in range(self.ppo_steps):
            # get traj from all states by actor and critic model
            action_pred = self.actor(self.states)
            value_pred = self.critic(self.states)
            action_prob = F.softmax(action_pred, dim = -1)
            move_dist = Categorical(action_prob)

            # get new log_p by using old actions
            new_log_p_actions = move_dist.log_prob(self.actions)
            policy_ratio = (new_log_p_actions - self.saved_actions).exp()
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.ppo_clip, max = 1.0 + self.ppo_clip) * advantages

            policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
            value_loss = F.smooth_l1_loss(returns, value_pred.squeeze(-1)).sum()

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return total_policy_loss/self.ppo_steps, total_value_loss/self.ppo_steps

    def in_train_eva(self):
        self.actor.eval()
        self.critic.eval()

        eva_rewards = []
        state = self.env.reset()
        done = False


        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_pred = self.actor(state)
                action_prob = F.softmax(action_pred, dim = -1)
            
            action = torch.argmax(action_prob, dim = -1)
            state, reward, done, _ = self.env.step(action.item())
            eva_rewards.append(reward)
        
        last_reward = np.sum(eva_rewards)
        self.eva_avg_reward = last_reward if not self.eva_avg_reward else self.eva_avg_reward * 0.9 + last_reward * 0.1

    def train(self):
        print('PG with PPO (Proximal Policy Optimization)')
        self.record_reward = []

        avg_reward = None
        self.eva_avg_reward = None
        out_train_reward = []
        out_test_reward = []
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                self.actor.train()
                self.critic.train()
                action = self.make_action(state)
                
                state, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)

            # update model
            self.states = torch.cat(self.states)
            self.actions = torch.cat(self.actions)
            self.saved_actions = torch.cat(self.saved_actions)
            self.values = torch.cat(self.values).squeeze(-1)

            policy_loss, value_loss = self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            # self.rewards = []
            # self.saved_actions = []
            self.record_reward.append(avg_reward)

            if epoch % self.display_freq == 0:
                # print(policy_loss, value_loss)
                self.in_train_eva()
                out_train_reward.append(avg_reward)
                out_test_reward.append(self.eva_avg_reward)
                print('Epochs: %d/%d | Avg reward: %f | Eva Avg reward: %f' %
                       (epoch, self.num_episodes, avg_reward, self.eva_avg_reward))
                
                np.save('pg_train_10_reward', out_train_reward)
                np.save('pg_test_10_reward', out_test_reward)

            if self.eva_avg_reward > REWARD_STOP: # to pass baseline, avg. reward > 50 is enough.
                self.in_train_eva()
                print('Epochs: %d/%d | Avg reward: %f | Eva Avg reward: %f'%
                       (epoch, self.num_episodes, avg_reward, self.eva_avg_reward))
                self.record_reward = np.array(self.record_reward)
                np.save('pg_reward', self.record_reward)
                self.save('pg_actor.cpt', 'pg_critic.cpt')
                self.in_train_eva()
                out_train_reward.append(avg_reward)
                out_test_reward.append(self.eva_avg_reward)
                np.save('pg_train_10_reward', out_train_reward)
                np.save('pg_test_10_reward', out_test_reward)
                break

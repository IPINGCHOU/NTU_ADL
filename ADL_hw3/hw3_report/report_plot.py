import numpy as np
import matplotlib.pyplot as plt
#%%
def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

#%%

test = [10,100,1000,10000]
fig, axs = plt.subplots(2,2, figsize=(15,12))
MA = 200
count = 0
for row in range(2):
    for col in range(2):
        dqn_reward = np.load('dqn_reward_'+str(test[count])+'.npy')
        dqn_reward = moving_average(dqn_reward, MA)
        axs[row][col].set_title('dqn reward with train_net update = ' + str(test[count]))
        axs[row][col].set_xlabel('iters after MA = ' + str(MA))
        axs[row][col].set_ylabel('reward after MA = ' + str(MA))
        axs[row][col].plot(range(len(dqn_reward)), dqn_reward)
        count += 1

#%%

test = [10,100,1000,10000]
fig, axs = plt.subplots(1,1, figsize=(8,6))
MA = 200
axs.set_title('dqn reward with train_net diff update')
axs.set_xlabel('iters after MA = ' + str(MA))
axs.set_ylabel('reward after MA = ' + str(MA))
for t in test:
    dqn_reward = np.load('dqn_reward_'+str(t)+'.npy')
    dqn_reward = moving_average(dqn_reward, MA)
    axs.plot(range(len(dqn_reward)), dqn_reward, label = str(t))
axs.legend(loc = 'upper left')
# %%

test = [1000]
fig, axs = plt.subplots(1,1, figsize=(8,6))
MA = 200
axs.set_title('dqn reward with train_net diff update')
axs.set_xlabel('iters after MA = ' + str(MA))
axs.set_ylabel('reward after MA = ' + str(MA))
for t in test:
    dqn_reward = np.load('dqn_reward_'+str(t)+'.npy')
    dqn_reward = moving_average(dqn_reward, MA)
    axs.plot(range(len(dqn_reward)), dqn_reward, label = 'ddqn ' + str(t))

duel_test = [1000]
for d in duel_test:
    dqn_reward = np.load('Duelingdqn_reward_'+str(d)+'.npy')
    dqn_reward = moving_average(dqn_reward, MA)
    axs.plot(range(len(dqn_reward)), dqn_reward, label = 'dueling ddqn' + str(d))
axs.legend(loc = 'upper left')

# %%

pg_reward = np.load('naive_pg_reward.npy')
ppo_reward = np.load('pg_reward.npy')
fig, axs = plt.subplots(1,1, figsize = (8,6))
axs.set_title('Naive PG and PPO reward, stop at Avg reward = 50')
axs.set_xlabel('steps')
axs.set_ylabel('reward')

axs.plot(range(len(pg_reward)), pg_reward, label = 'Naive PG')
axs.plot(range(len(ppo_reward)), ppo_reward, label = 'PPO')
axs.axhline(y = 50, color = 'r')
axs.legend(loc = 'upper left')

# %%


# %%
ppo_train = np.load('pg_train_10_reward.npy')
ppo_test = np.load('pg_test_10_reward.npy')
fig, axs = plt.subplots(1,1, figsize = (8,6))
axs.set_title('PPO reward, test with max probs steps, stop at train Avg reward = 200')
axs.set_xlabel('steps')
axs.set_ylabel('reward')

axs.plot(range(len(ppo_train)), ppo_train, label = 'Train')
axs.plot(range(len(ppo_test)), ppo_test, label = 'Test')
axs.axhline(y = 100, color = 'r')
axs.legend(loc = 'upper left')

# %%

fig, axs = plt.subplots(1,1, figsize = (8,6))
MA = 200
END = 550
axs.set_title('Double DQN with Dueling Double DQN, target net update = 1000')
axs.set_xlabel('episode')
axs.set_ylabel('rewards')
dqn_reward = np.load('dqn_reward_'+str(1000)+'.npy')
dqn_reward = moving_average(dqn_reward, MA)
dqn_reward = dqn_reward[:END]
axs.plot(range(len(dqn_reward)), dqn_reward, label = 'double dqn' + str(1000))

dqn_reward = np.load('small_dueling_dqn_reward_'+str(1000)+'.npy')
dqn_reward = moving_average(dqn_reward, MA)
dqn_reward = dqn_reward[:END]
axs.plot(range(len(dqn_reward)), dqn_reward, label = 'dueling double dqn' + str(1000))

axs.legend(loc = 'upper left')

# %%

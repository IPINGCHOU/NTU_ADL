import numpy as np
import matplotlib.pyplot as plt
#%%
def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


#%%
# pq reward plot
pq_reward = np.load('pg_reward.npy')
MA = 100
pq_reward = moving_average(pq_reward, MA)
plt.figure(figsize=(8,6))
plt.title('PQ reward with MA = ' + str(MA))
plt.xlabel('iters after MA')
plt.ylabel('reward after MA')
plt.plot(range(len(pq_reward)), pq_reward)
plt.show()
#%%

dqn_reward = np.load('dqn_reward.npy')
MA = 100000
dqn_reward = moving_average(dqn_reward, MA)
plt.figure(figsize=(8,6))
plt.title('dqn reward with MA = ' + str(MA))
plt.xlabel('iters after MA')
plt.ylabel('reward after MA')
plt.plot(range(len(dqn_reward)), dqn_reward)
plt.show()

# %%

# %%

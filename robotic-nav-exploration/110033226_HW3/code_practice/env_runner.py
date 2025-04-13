import torch
import numpy as np
from collections import deque

# 折扣回報計算 (Compute discounted return)
def compute_discounted_return(rewards, dones, last_values, last_dones, gamma=0.99):
    returns = np.zeros_like(rewards)
    n_step = len(rewards)

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            returns[t] = rewards[t] + gamma * last_values * (1.0 - last_dones)
        else:
            returns[t] = rewards[t] + gamma * returns[t + 1] * (1.0 - dones[t + 1])
    return returns

# 廣義優勢估計 (Compute Generalized Advantage Estimation, GAE)
def compute_gae(rewards, values, dones, last_values, last_dones, gamma=0.99, lamb=0.95):
    advs = np.zeros_like(rewards)
    n_step = len(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            next_nonterminal = 1.0 - last_dones
            next_values = last_values
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        advs[t] = last_gae_lam = delta + gamma * lamb * next_nonterminal * last_gae_lam

    return advs + values

# 多環境資料收集器 (Runner for multiple environments)
class EnvRunner:
    # 建構子 Constructor
    def __init__(self, env, s_dim, a_dim, n_step=5, gamma=0.99, lamb=0.95, device='cpu'):
        self.env = env
        self.n_env = env.n_env
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_step = n_step
        self.gamma = gamma
        self.lamb = lamb
        self.device = device

        # 最近一次狀態與終止狀態 (last states and dones)
        self.states = self.env.reset()
        self.dones = np.ones((self.n_env), dtype=bool)

        # 批次儲存區 (Memory buffers for each time step)
        self.mb_states = np.zeros((self.n_step, self.n_env, self.s_dim), dtype=np.float32)
        self.mb_actions = np.zeros((self.n_step, self.n_env, self.a_dim), dtype=np.float32)
        self.mb_values = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_rewards = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_dones = np.zeros((self.n_step, self.n_env), dtype=bool)

        # 回報與執行長度紀錄器 (Reward and length tracking)
        self.total_rewards = np.zeros((self.n_env), dtype=np.float32)
        self.total_len = np.zeros((self.n_env), dtype=np.int32)
        self.reward_buf = deque(maxlen=100)
        self.len_buf = deque(maxlen=100)

	#TODO 3: Run a step to collect data
    # 執行 n 步收集資料 (Run n steps to collect data)
    def run(self, policy_net, value_net):
        # 第一步：執行 n_step 步
        for step in range(self.n_step):
            # 將目前狀態轉成 tensor，並移至裝置
            state_tensor = torch.from_numpy(self.states).float().to(self.device)
            
            # 取得動作與對數機率 (Get action and log probability)
            actions, a_logps = policy_net(state_tensor, deterministic=False)

            # 取得狀態價值 (Get value predictions)
            values = value_net(state_tensor)

            # 儲存資料 (Store collected data)
            self.mb_states[step, :] = self.states
            self.mb_dones[step, :] = self.dones
            self.mb_actions[step, :] = actions.cpu().numpy()
            self.mb_a_logps[step, :] = a_logps.detach().cpu().numpy()
            self.mb_values[step, :] = values.detach().cpu().numpy()

            # 與環境互動 (Interact with environment)
            self.states, rewards, self.dones, info = self.env.step(actions.cpu().numpy())
            self.mb_rewards[step, :] = rewards

        # 最後一步的 value（估算下一狀態）
        last_values = value_net(torch.from_numpy(self.states).float().to(self.device)).cpu().numpy()

        # 紀錄當前回合回報與長度 (Log current episode stats)
        self.record()

        # 第二步：計算 returns（含 GAE）
        mb_returns = compute_gae(self.mb_rewards, self.mb_values, self.mb_dones, last_values, self.dones, self.gamma, self.lamb)

        return self.mb_states.reshape(self.n_step * self.n_env, self.s_dim), \
               self.mb_actions.reshape(self.n_step * self.n_env, self.a_dim), \
               self.mb_a_logps.flatten(), \
               self.mb_values.flatten(), \
               mb_returns.flatten()

    # 紀錄回報與長度 (Log reward and episode length)
    def record(self):
        for i in range(self.n_step):
            for j in range(self.n_env):
                if self.mb_dones[i, j]:
                    self.reward_buf.append(self.total_rewards[j] + self.mb_rewards[i, j])
                    self.len_buf.append(self.total_len[j] + 1)
                    self.total_rewards[j] = 0
                    self.total_len[j] = 0
                else:
                    self.total_rewards[j] += self.mb_rewards[i, j]
                    self.total_len[j] += 1

    # 取得最近表現統計 (Return mean reward, std, and episode length)
    def get_performance(self):
        if len(self.reward_buf) == 0:
            mean_return = 0
            std_return = 0
        else:
            mean_return = np.mean(self.reward_buf)
            std_return = np.std(self.reward_buf)

        if len(self.len_buf) == 0:
            mean_len = 0
        else:
            mean_len = np.mean(self.len_buf)

        return mean_return, std_return, mean_len
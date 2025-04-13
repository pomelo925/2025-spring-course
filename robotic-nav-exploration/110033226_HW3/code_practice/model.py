import numpy as np
import torch
import torch.nn as nn

# 權重初始化 (Weight initialization)
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# 固定均值與標準差的常態分佈 (Normal distribution with fixed mean and std)
class FixedNormal(torch.distributions.Normal):
	# 對數機率 (Log-probability)
	def log_probs(self, actions):
		return super().log_prob(actions).sum(-1)

	# 熵 (Entropy)
	def entropy(self):
		return super().entropy().sum(-1)

	# 取分佈的眾數 (Mode of the distribution)
	def mode(self):
		return self.mean

# 對角高斯分佈 (Diagonal Gaussian distribution)
class DiagGaussian(nn.Module):
	# 建構子 (Constructor)
	def __init__(self, inp_dim, out_dim, std=0.5):
		super(DiagGaussian, self).__init__()

		# 初始化函數 (Initialization function)
		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,           # 正交初始化 (Orthogonal initialization)
			lambda x: nn.init.constant_(x, 0)  # 偏差初始化為0 (Bias initialized to 0)
		)
		self.fc_mean = init_(nn.Linear(inp_dim, out_dim))  # 平均輸出層 (Mean output layer)
		self.std = torch.full((out_dim,), std)             # 固定標準差 (Fixed std)

	# 前向傳遞 (Forward pass)
	def forward(self, x):
		mean = self.fc_mean(x)
		return FixedNormal(mean, self.std.to(x.device))

# 策略網路 (Policy Network)
class PolicyNet(nn.Module):
	# 建構子 (Constructor)
	def __init__(self, s_dim, a_dim, std=0.5):
		super(PolicyNet, self).__init__()

		# 初始化函數 (Initialization function)
		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			nn.init.calculate_gain('relu')  # 根據 ReLU 計算 gain (Gain for ReLU)
		)

		# 策略網路主幹 (Policy network architecture)
		self.main = nn.Sequential(
			init_(nn.Linear(s_dim, 128)),  # 輸入層 (Input layer)
			nn.ReLU(),                    # ReLU 激活函數 (ReLU activation)
			init_(nn.Linear(128, 128)),     # 隱藏層 (Hidden layer)
			nn.ReLU()
		)
		self.dist = DiagGaussian(128, a_dim, std=std)  # 動作分佈 (Action distribution)

	# 前向傳遞 (Forward pass)
	def forward(self, state, deterministic=False):
		feature = self.main(state)
		dist    = self.dist(feature)

		if deterministic:
			action = dist.mode()         # 決定性模式取眾數 (Deterministic: take mode)
		else:
			action = dist.sample()       # 隨機取樣 (Stochastic sampling)

		return action, dist.log_probs(action)

	# 輸出動作 (Output action)
	def action_step(self, state, deterministic=True):
		feature = self.main(state)
		dist    = self.dist(feature)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		return action

	# 評估對數機率與熵 (Evaluate log-probs & entropy)
	def evaluate(self, state, action):
		feature = self.main(state)
		dist    = self.dist(feature)
		return dist.log_probs(action), dist.entropy()

# 價值網路 (Value Network)
class ValueNet(nn.Module):
	# 建構子 (Constructor)
	def __init__(self, s_dim):
		super(ValueNet, self).__init__()

		# 初始化函數 (Initialization function)
		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			nn.init.calculate_gain('relu')
		)

		# 價值網路主幹 (Value network architecture)
		self.main = nn.Sequential(
			init_(nn.Linear(s_dim,128)),  # 輸入層 (Input layer)
			nn.ReLU(),
			init_(nn.Linear(128, 128)),     # 隱藏層 (Hidden layer)
			nn.ReLU(),
			init_(nn.Linear(128, 1))       # 輸出一個值 (Output one value)
		)

	# 前向傳遞 (Forward pass)
	def forward(self, state):
		return self.main(state)[:, 0]  # 回傳值向量的第一維 (Return scalar value)
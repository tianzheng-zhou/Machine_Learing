# 导入必要的库
import numpy as np
import random
from numba import jit
import torch
import torch.nn as nn
import multiprocessing
from typing import Callable
import args

global space  # 全局状态空间
space = []  # 存储所有状态块的列表


class block:
    """表示环境中的基本状态块"""

    def __init__(self, actions: list):
        self.position = 0  # 状态位置索引
        self.state_value = 0  # 状态价值
        self.state_reward = 0  # 状态即时奖励
        self.actions = actions  # 可执行的动作列表
        self.action_values = np.array([0] * len(actions))  # 各动作的价值估计


class set():
    def __init__(self, blocks: list, *maxlen):
        self.blocks = list(blocks)  # 所有状态块的集合
        self.maxlen = list(maxlen)  # 各维度的最大长度
        prod = 1
        for i in maxlen:  # 计算总状态数
            prod *= i
        self.length = prod
        cnt = 0
        for i in self.blocks:  # 为每个块分配位置索引
            i.postion = cnt
            cnt += 1

    # 以下方法实现容器的基本操作
    def __len__(self):
        return self.length

    def __iter__(self):
        return self.blocks.__iter__()

    def __getitem__(self, index):
        return self.blocks[index]

    def getitem(self, *keys):
        """多维索引访问方法"""
        position = 0
        base = 1
        for i in range(len(self.maxlen) - 1, -1, -1):  # 计算线性索引
            position += keys[i] * base
            base *= self.maxlen[i]
        return self.blocks[position]

    def get_index(self, *keys):
        """获取多维索引对应的线性索引"""
        position = 0
        base = 1
        for i in range(len(self.maxlen) - 1, -1, -1):
            position += keys[i] * base
            base *= self.maxlen[i]
        return position


class action:
    """表示可执行的动作及其转移特性"""

    def __init__(self, next_state: list, next_state_p: np.ndarray):
        self.action_value = 0  # 动作价值估计
        self.action_reward = 0  # 执行动作的即时奖励
        self.visit_cnt = 0  # 动作被访问次数
        self.next_state = next_state  # 可能的下个状态列表
        self.next_state_p = next_state_p  # 状态转移概率分布
        self.direction = []  # 动作方向信息（可选）

    # self.next_state_p_expected=np.array([1/len(next_state)]*len(next_state)) #the expected position after some action used for questions that the next state is not settled after a certain action
    def __iter__(self):
        return self

    def __next__(self) -> int:  #giving next state by taking this action
        total = np.sum(self.next_state_p)
        x = random.random() * total
        for i in range(len(self.next_state_p)):
            x -= self.next_state_p[i]
            if x <= 0:
                return self.next_state[i]
        return self.next_state[0]

    def update(self, ret, a: Callable[[int, ], float] = lambda x: 1 / x):  # 'a' is the conerge coiffiecient in RM
        self.visit_cnt += 1
        self.action_value = self.action_value - a(self.visit_cnt) * (self.action_value - ret)


class policy:
    """决策策略基类"""

    def __init__(self, principle: Callable[[np.ndarray, ], np.ndarray]):
        self.principle = principle  # 策略选择函数（如epsilon-greedy）

    def __call__(self, state: block) -> np.ndarray:
        """获取动作选择概率分布"""
        values = list(map(lambda x: x.action_value, state.actions))
        return self.principle(values)

    def choice(self, state: block) -> action:  # return the action the policy choose
        probability = self.__call__(state)
        total = probability.sum()
        x = total * random.random()
        for i in range(len(probability)):
            x -= probability[i]
            if x <= 0:
                return i
        return 0


class episode:
    """表示一个完整的交互轨迹"""

    def __init__(self, state: int, policy: policy, step: int):
        self.track = []  # 存储（状态，动作）序列
        self.now_state = state  # 当前状态
        self.policy = policy  # 使用的策略

        # 处理不同步长模式
        if step < 0:  # 持续运行直到达到终止状态
            while space[self.now_state].state_reward <= 0:
                a = policy.choice(space[self.now_state])
                #a=self.act(policy)
                self.track.append((self.now_state, a))
                self.now_state = next(space[self.now_state].actions[a])
            return

        # 有限步长模式
        for i in range(step):
            a = policy.choice(space[self.now_state])
            #a=self.act(policy)
            self.track.append((self.now_state, a))
            if space[self.now_state].state_reward > 0:  # 遇到终止状态提前结束
                return
            self.now_state = next(space[self.now_state].actions[a])

    def __iter__(self) -> list:
        return self.track

    def act(self, policy: policy) -> action:  # return the action the policy choose  ,same as choice in class policy
        return policy.choice()


class QN(nn.Module):
    """Q网络神经网络实现"""

    def __init__(self, state_dim, action_dim):
        super(QN, self).__init__()
        # 网络层定义
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, action_dim)
        torch.nn.init.uniform_(self.fc1.weight, -0.15, 0.15)
        torch.nn.init.uniform_(self.fc2.weight, -0.15, 0.15)
        torch.nn.init.uniform_(self.fc3.weight, -0.15, 0.15)
        torch.nn.init.uniform_(self.fc4.weight, -0.15, 0.15)
        torch.nn.init.uniform_(self.fc5.weight, -0.15, 0.15)
        torch.nn.init.uniform_(self.fc6.weight, -0.15, 0.15)
        self.function = nn.functional.leaky_relu
        #self.inp,self.out=multiprocessing.Pipe(True)
        #self.inp=inp_queue
        self.criterion = torch.nn.SmoothL1Loss()  #reduction='mean')
        self.optimizer = torch.optim.SGD(self.parameters(), args.lr_rate)

    def forward(self, x):
        """前向传播过程"""
        x = self.function(self.fc1(x))
        x = self.function(self.fc2(x))
        x = torch.nn.functional.sigmoid(self.fc3(x))  # 中间层使用sigmoid
        x = self.function(self.fc4(x))
        x = self.function(self.fc5(x))
        return self.fc6(x)

    def train(self, mini_batch):
        """训练方法（使用经验回放）"""
        batch_length = len(mini_batch)
        sa, action_value = zip(*mini_batch)
        # next_states = np.array(next_states)
        pred = torch.zeros(batch_length, device=args.device)
        for epoch in range(batch_length):
            torch.nn.utils.clip_grad_value_(self.parameters(), 100)  # 梯度裁剪
            # print(f"TN sa:{sa[epoch]}")
            # print(f"TN pred:{pred}")
            # print(f"TN action_value{action_value[epoch]}")
            # print(f"TN av tensor{torch.tensor((action_value[epoch],)).to(args.device)}")
            pred[epoch] = self.forward(torch.tensor(sa[epoch]).to(args.device))
            # print(f"TN sa:{sa[epoch]}")
            # print(f"TN pred:{pred}")
            torch.nn.utils.clip_grad_value_(self.parameters(), 100)
            # print(f"TN av tensor{torch.tensor((action_value[epoch],)).to(args.device)}")
            loss = self.criterion(pred, torch.tensor((action_value)).to(args.device))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), 100)
            self.optimizer.step()
            print(f"loss:{loss}")
            return 1

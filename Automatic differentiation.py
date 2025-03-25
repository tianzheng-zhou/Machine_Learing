import numpy as np


def activate_function(x):
    return 1 / (1 + np.exp(-x))  # sigmoid函数实现


def d_activate_function(x):
    s = 1 / (1 + np.exp(-x))  # 复用sigmoid函数计算
    return s * (1 - s)  # 导数公式 σ'(x) = σ(x)(1-σ(x))


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data  # 节点存储的数值
        self.grad = 0.0  # 梯度值（初始为0）
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self.grad_fn = None  # 生成该节点的运算（如加法、乘法）
        self.parents = []  # 输入节点列表（父节点，构成计算图的边）

        self.weight = None  # 权重
        self.bias = 0  # 偏置

    def backward(self, other):
        if self.grad_fn == "forward":
            other.grad = self._activate(other)
            other.requires_grad = True

    def _activate(self, other):
        # 仅仅计算节点的梯度 而不是偏置或者权重

        temp_grad = self.data * d_activate_function(other.data)
        temp_grad *= other.weight

        return temp_grad

    def forward(self, other):
        # 前向传播，用于生成计算图
        self.parents.append(other)
        self.data = activate_function(other.data*other.weight + other.bias)
        other.grad_fn = "forward"

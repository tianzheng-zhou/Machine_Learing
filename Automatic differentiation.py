import numpy as np


def activate_function(x):
    return 1 / (1 + np.exp(-x))  # sigmoid函数实现


def d_activate_function(x):
    s = 1 / (1 + np.exp(-x))  # 复用sigmoid函数计算
    return s * (1 - s)  # 导数公式 σ'(x) = σ(x)(1-σ(x))


class Tensor:
    def __init__(self, data: np.array, requires_grad=False):
        self.data = data  # 节点存储的数值
        self.grad = 0.0  # 梯度值（初始为0）
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self.op = None  # 生成该节点的运算（如加法、乘法）
        self.parents = []  # 输入节点列表（父节点，构成计算图的边）

    def __add__(self, other):
        # 加法运算
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, requires_grad=True)
            out.op = 'add'
            out.parents = [self, other]
            return out
        else:  # 估计是用不到了
            out = Tensor(self.data + other, requires_grad=True)
            out.op = 'add'
            out.parents = [self]
            return out

    def add_forward(self, other):
        return self + other

    def add_backward(self, grad):
        pass

    def __mul__(self, other):
        # 乘法运算
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, requires_grad=True)
            out.op = 'mul'
            out.parents = [self, other]
            return out
        else:
            out = Tensor(self.data * other, requires_grad=True)
            out.op = 'mul'
            out.parents = [self]
            return out

    def mul_forward(self, other):
        return self * other

    def mul_backward(self, grad):
        pass

    def activate_forward(self):
        out = Tensor(activate_function(self.data), requires_grad=True)
        out.op = 'activate'
        out.parents = [self]
        return out

    def activate_backward(self, grad):
        pass

    def dot_forward(self, other):
        out = Tensor(np.dot(self.data, other.data), requires_grad=True)
        out.op = 'dot'
        out.parents = [self, other]
        return out

    def dot_backward(self, grad):
        pass

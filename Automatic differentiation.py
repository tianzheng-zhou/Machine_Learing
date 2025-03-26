import numpy as np
import struct


class Tensor:
    def __init__(self, data: np.array, requires_grad=False):
        self.data = data  # 节点存储的数值
        self.grad = None  # 梯度值（初始为None）
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self.op = None  # 生成该节点的运算（如加法、乘法）
        self.parents = []  # 输入节点列表（父节点，构成计算图的边）

    @staticmethod
    def activate_function(x):
        return 1 / (1 + np.exp(-x))  # sigmoid函数实现

    @staticmethod
    def d_activate_function(x):
        s = 1 / (1 + np.exp(-x))  # 复用sigmoid函数计算
        return s * (1 - s)  # 导数公式 σ'(x) = σ(x)(1-σ(x))

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

    def add_backward(self):
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

    def mul_backward(self):
        if self.op == "mul":
            if self.parents[0].grad is None:
                self.parents[0].grad = self.grad * self.parents[1].data
            else:
                self.parents[0].grad += self.grad * self.parents[1].data
            if self.parents[1].grad is None:
                self.parents[1].grad = self.grad * self.parents[0].data
            else:
                self.parents[1].grad += self.grad * self.parents[0].data
        else:
            print("Error: mul_backward only works for mul operation.")

    def activate_forward(self):
        out = Tensor(self.activate_function(self.data), requires_grad=True)
        out.op = 'activate'
        out.parents = [self]
        return out

    def activate_backward(self):
        if self.op == "activate":
            if self.parents[0].grad is None:
                self.parents[0].grad = self.grad * self.d_activate_function(self.parents[0].data)
            else:
                self.parents[0].grad += self.grad * self.d_activate_function(self.parents[0].data)
        else:
            print("Error: activate_backward only works for activate operation.")

    def dot_forward(self, other):
        # 矩阵在左
        out = Tensor(np.dot(self.data, other.data), requires_grad=True)
        out.op = 'dot'
        out.parents = [self, other]
        return out

    def dot_backward(self):
        if self.op == "dot":
            if self.parents[0].grad is None:
                self.parents[0].grad = np.dot(self.grad, self.parents[1].data.T)
            else:
                self.parents[0].grad += np.dot(self.grad, self.parents[1].data.T)

            if self.parents[1].grad is None:
                self.parents[1].grad = np.dot(self.parents[0].data.T, self.grad)
            else:
                self.parents[1].grad += np.dot(self.parents[0].data.T, self.grad)
        else:
            print("Error: dot_backward only works for dot operation.")

class TensorNetwork:
    def __init__(self, depth, layer_size: tuple):
        # 注意：layer_size最后一层应当为10
        # 初始化网络
        self.input = np.ones(784)  # 输入向量，784维（28x28图像展开）
        self.cost = 0  # 损失值
        # 定义隐藏层层数
        self.depth = depth  # 网络深度（层数）
        self.layer_size = layer_size  # 每层的神经元数量
        # layers作为二维数组 存储隐藏层 内部元素为np数组
        self.layers = []  # 存储每一层的激活值
        # 初始化每一层的激活值为0


def read_images(filepath):
    # 读取MNIST图像文件
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 0x00000803, "Invalid image file format"
        # 一次性读取所有图像数据并转换为numpy数组
        images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        return images.reshape(num, rows, cols)  # 转换为三维数组 (样本数, 行, 列)


def read_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 0x00000801, "Invalid label file format"
        # 读取所有标签数据并转换为numpy数组
        return np.frombuffer(f.read(num), dtype=np.uint8)

if __name__ == '__main__':
    DEBUG = True
    # 示例用法
    train_images = read_images('data\\train-images.idx3-ubyte')
    train_labels = read_labels('data\\train-labels.idx1-ubyte')

    test_images = read_images('data\\t10k-images.idx3-ubyte')
    test_labels = read_labels('data\\t10k-labels.idx1-ubyte')


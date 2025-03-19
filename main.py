import numpy as np
import struct
import matplotlib.pyplot as plt


def read_images(filepath):
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


# 示例用法
train_images = read_images('data\\train-images.idx3-ubyte')
train_labels = read_labels('data\\train-labels.idx1-ubyte')

# 访问第0张图像和标签
print("Label:", train_labels[0])
for i in range(28):
    print(f"Pixel values (第{i}行):", train_images[0][i])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Network:
    def __init__(self, depth, layer_size: tuple):
        # 初始化网络
        self.input = np.zeros(784)
        self.input2d = np.zeros((28, 28))
        self.output = np.zeros(10)

        # 定义隐藏层层数
        self.depth = depth
        self.layer_size = layer_size

        # layers作为二维数组 存储隐藏层 内部元素为np数组
        self.layers = []

        for _ in range(depth):
            self.layers.append(np.zeros(layer_size[_], dtype=float))

        self.weights = []
        self.biases = []
        for _ in range(depth):
            self.weights.append(np.zeros((layer_size[_],784), dtype=float))
            self.biases.append(np.zeros(layer_size[_], dtype=float))

    def forward(self, input: np.ndarray):

        self.input = input

        # 处理第一层神经
        # 乘上权重
        for j in range(self.layer_size[0]):
            for i in range(784):
                self.layers[0] = np.dot(self.weights[0], self.input)

            self.layers[0][j] += self.biases[0][j]  # 加上bias
        # activate by sigmoid
        self.layers[0] = sigmoid(self.layers[0])

        # 处理后面几层神经
        for i in range(1,self.depth):
            self.layers[i] = self.layers[i-1] * self.weights[i] + self.biases[i]

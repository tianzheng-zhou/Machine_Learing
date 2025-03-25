import numpy as np
import struct


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    # return np.maximum(0, x)
    # return np.where(x > 0, x, 0.01 * x)


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
    # return (x > 0).astype(float)
    # return np.where(x > 0, 1, 0.01)


class Network:
    def __init__(self, depth, layer_size: tuple):
        # 注意：layer_size最后一层应当为10
        # 初始化网络
        self.input = np.ones(784)  # 输入向量，784维（28x28图像展开）
        self.input2d = np.zeros((28, 28))  # 二维输入，用于可视化

        self.cost = 0  # 损失值

        # 定义隐藏层层数
        self.depth = depth  # 网络深度（层数）
        self.layer_size = layer_size  # 每层的神经元数量

        # layers作为二维数组 存储隐藏层 内部元素为np数组
        self.layers = []  # 存储每一层的激活值

        # 初始化每一层的激活值为0
        for _ in range(depth):
            self.layers.append(np.zeros(layer_size[_], dtype=float))

        # 初始化权重矩阵和偏置矩阵
        self.weights = []  # 权重矩阵列表
        self.biases = []  # 偏置向量列表

        # 替换初始化代码
        # 下面这一段是ai写的，不要动
        # 使用He初始化方法初始化第一层权重
        self.weights.append(np.random.randn(layer_size[0], 784) * np.sqrt(2. / 784))
        self.biases.append(np.zeros(layer_size[0]))  # 偏置初始化为0
        # 使用He初始化方法初始化后续层权重
        for _ in range(1, depth):
            prev_size = layer_size[_ - 1]  # 前一层的神经元数量
            self.weights.append(np.random.randn(layer_size[_], prev_size) * np.sqrt(2. / prev_size))
            self.biases.append(np.zeros(layer_size[_]))

        '''
        自己写的 全1初始化会导致发生我们不希望的事情
        self.weights.append(np.ones((layer_size[0], 784), dtype=np.float64))
        self.biases.append(np.ones(layer_size[0], dtype=np.float64))
        for _ in range(1, depth):
            self.weights.append(np.ones((layer_size[_], layer_size[_-1]), dtype=np.float64))
            self.biases.append(np.ones(layer_size[_], dtype=np.float64))'''

        # 初始化梯度临时存储
        self.gradient_temp = []  # 存储每一层的梯度
        for _ in range(depth):
            self.gradient_temp.append(np.ones(layer_size[_], dtype=np.float64))

        # 初始化权重梯度存储
        self.gradient_for_weight = []  # 存储每一层权重的梯度
        self.gradient_for_weight.append(np.ones((layer_size[0], 784), dtype=np.float64))
        for _ in range(1, depth):
            self.gradient_for_weight.append(np.ones((layer_size[_], layer_size[_ - 1]), dtype=np.float64))
        pass

    def forward(self, input: np.ndarray):
        # 前向传播函数
        self.input = input  # 保存输入数据

        # 处理第一层神经
        # 乘上权重
        self.layers[0] = np.dot(self.weights[0], self.input)  # 权重矩阵点乘输入向量

        # 使用sigmoid激活函数
        self.layers[0] = sigmoid(self.layers[0])

        # 处理后面几层神经
        for i in range(1, self.depth):
            # 计算当前层的加权和并加上偏置
            self.layers[i] = np.dot(self.weights[i], self.layers[i - 1]) + self.biases[i]
            # 使用sigmoid激活函数
            self.layers[i] = sigmoid(self.layers[i])

        # 返回最后的输出层 形式为numpy数组
        return self.layers[-1]

    def calculate_cost(self, target: np.ndarray):
        # target: one hot编码的目标
        self.cost = np.sum(np.square(target - self.layers[-1]))
        return self.cost

    def backward(self, target: np.ndarray, learning_rate: float):
        # 反向传播算法实现
        # 计算输出层梯度：损失函数导数 * 激活函数导数
        self.gradient_temp[-1] = (2 * (self.layers[-1] - target) *
                                  d_sigmoid(np.dot(self.weights[-1], self.layers[-2]) +
                                            self.biases[-1]))

        # 计算输出层权重梯度：梯度向量外积前一层的激活值
        self.gradient_for_weight[-1] = np.dot(self.gradient_temp[-1].reshape(-1, 1),
                                              self.layers[-2].reshape(1, -1))

        # 更新输出层权重和偏置
        self.weights[-1] -= learning_rate * self.gradient_for_weight[-1]
        self.biases[-1] -= learning_rate * self.gradient_temp[-1]

        # 反向传播至各隐藏层（从倒数第二层开始）
        for i in range(self.depth - 2, -1, -1):
            # 处理输入层（第0层）的特殊情况
            if i == 0:

                # 遗留的屎山 改不下去就重写了
                '''self.gradient_temp[i] = np.dot(self.gradient_temp[i + 1], (np.dot(self.weights[i + 1].transpose(),

                                                                           d_sigmoid(np.dot(self.weights[i],
                                                                                            self.input)
                                                                                     + self.biases[i])
                                                                                  )
                                                                           ))'''
                self.gradient_temp[i] = (
                        np.dot(self.weights[i + 1].transpose(), self.gradient_temp[i + 1]) *
                        d_sigmoid(np.dot(self.weights[i], self.input) + self.biases[i])
                )
                # 计算输入层权重梯度：使用原始输入数据
                self.gradient_for_weight[i] = np.dot(self.gradient_temp[i].reshape(-1, 1), self.input.reshape(1, -1))

            # 处理隐藏层
            else:

                # 遗留的屎山
                '''self.gradient_temp[i] = np.dot(self.gradient_temp[i + 1], (np.dot(self.weights[i + 1].transpose() ,
                                                                           d_sigmoid(np.dot(self.weights[i],
                                                                                            self.layers[i - 1]) +
                                                                                     self.biases[i]))))'''
                self.gradient_temp[i] = (
                        np.dot(self.weights[i + 1].transpose(), self.gradient_temp[i + 1]) *
                        d_sigmoid(np.dot(self.weights[i], self.layers[i - 1]) + self.biases[i])
                )
                # 计算当前层权重梯度：使用前一层激活值
                self.gradient_for_weight[i] = np.dot(self.gradient_temp[i].reshape(-1, 1),
                                                     self.layers[i - 1].reshape(1, -1))

            # 更新当前层权重和偏置
            self.weights[i] -= learning_rate * self.gradient_for_weight[i]
            self.biases[i] -= learning_rate * self.gradient_temp[i]

            pass

        pass


if __name__ == '__main__':
    DEBUG = True
    # 示例用法
    train_images = read_images('data\\train-images.idx3-ubyte')
    train_labels = read_labels('data\\train-labels.idx1-ubyte')
    network = Network(depth=3, layer_size=(50, 40, 10))

    test_images = read_images('data\\t10k-images.idx3-ubyte')
    test_labels = read_labels('data\\t10k-labels.idx1-ubyte')

    for i in range(60000):
        one_hot = np.zeros(10)
        one_hot[train_labels[i]] = 1
        temp = []
        for j in range(28):
            temp += list(train_images[i][j])

        # 用于debug
        if DEBUG:
            print(network.forward(np.array(temp) / 255))
            print(network.calculate_cost(np.array(one_hot)))
        else:
            network.forward(np.array(temp) / 255)

        network.backward(np.array(one_hot), 0.1)

    count = 0
    for i in range(10000):
        one_hot = np.zeros(10)
        one_hot[test_labels[i]] = 1
        temp = []
        for j in range(28):
            temp += list(test_images[i][j])

        if np.argmax(network.forward(np.array(temp))) == test_labels[i]:
            count += 1

    print(count)

'''
    # 访问第0张图像和标签
    print("Label:", train_labels[0])
    for i in range(28):
        print(f"Pixel values (第{i}行):", train_images[0][i])

'''

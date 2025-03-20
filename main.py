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

def sigmoid(x):
    #return 1.0 / (1.0 + np.exp(-x))
    #return np.maximum(0, x)
    return np.where(x > 0, x, 0.01 * x)


def d_sigmoid(x):
    # return sigmoid(x) * (1 - sigmoid(x))
    #return (x > 0).astype(float)
    return np.where(x > 0, 1, 0.01)

class Network:
    def __init__(self, depth, layer_size: tuple):
        # 注意：layer_size最后一层应当为10
        # 初始化网络
        self.input = np.ones(784)
        self.input2d = np.zeros((28, 28))

        self.cost = 0

        # 定义隐藏层层数
        self.depth = depth
        self.layer_size = layer_size

        # layers作为二维数组 存储隐藏层 内部元素为np数组
        self.layers = []

        for _ in range(depth):
            self.layers.append(np.zeros(layer_size[_], dtype=float))

        # 初始化权重矩阵和偏置矩阵
        self.weights = []
        self.biases = []

        self.weights.append(np.ones((layer_size[0], 784), dtype=np.float64))
        self.biases.append(np.ones(layer_size[0], dtype=np.float64))
        for _ in range(1, depth):
            self.weights.append(np.ones((layer_size[_], layer_size[_-1]), dtype=np.float64))
            self.biases.append(np.ones(layer_size[_], dtype=np.float64))

        self.gradient_temp = []
        for _ in range(depth):
            self.gradient_temp.append(np.ones(layer_size[_], dtype=np.float64))

        self.gradient_for_weight = []
        self.gradient_for_weight.append(np.ones((layer_size[0], 784), dtype=np.float64))
        for _ in range(1,depth):
            self.gradient_for_weight.append(np.ones((layer_size[_], layer_size[_-1]), dtype=np.float64))
        pass

    def forward(self, input: np.ndarray):

        self.input = input

        # 处理第一层神经
        # 乘上权重
        self.layers[0] = np.dot(self.weights[0], self.input)  # 权重矩阵点乘输入向量

        # activate by sigmoid
        self.layers[0] = sigmoid(self.layers[0])

        # 处理后面几层神经
        for i in range(1, self.depth):
            self.layers[i] = np.dot(self.weights[i], self.layers[i - 1]) + self.biases[i]
            self.layers[i] = sigmoid(self.layers[i])

        return self.layers[-1]  # 返回最后的输出层 形式为numpy数组

    def calculate_cost(self, target: np.ndarray):
        # target: one hot编码的目标
        self.cost = np.sum(np.square(target - self.layers[-1]))
        return self.cost

    def backward(self, target: np.ndarray, learning_rate: float):
        # 反向传播
        self.gradient_temp[-1] = (2 * (self.layers[-1] - target)*
                                  d_sigmoid(np.dot(self.weights[-1], self.layers[-2]) +
                                                 self.biases[-1]))

        self.gradient_for_weight[-1] = np.dot(self.gradient_temp[-1].reshape(-1,1),
                                              self.layers[-2].reshape(1,-1))

        self.weights[-1] -= learning_rate * self.gradient_for_weight[-1]
        self.biases[-1] -= learning_rate * self.gradient_temp[-1]


        for i in range(self.depth - 2, -1, -1):

            if i==0:

                '''self.gradient_temp[i] = np.dot(self.gradient_temp[i + 1], (np.dot(self.weights[i + 1].transpose(),

                                                                           d_sigmoid(np.dot(self.weights[i],
                                                                                            self.input)
                                                                                     + self.biases[i])
                                                                                  )
                                                                           ))'''
                self.gradient_temp[i] = (
                    np.dot(self.weights[i+1].transpose(), self.gradient_temp[i + 1])*
                    d_sigmoid(np.dot(self.weights[i], self.input) + self.biases[i])
                )

                self.gradient_for_weight[i] = np.dot(self.gradient_temp[i].reshape(-1,1), self.input.reshape(1,-1))

            else:

                '''self.gradient_temp[i] = np.dot(self.gradient_temp[i + 1], (np.dot(self.weights[i + 1].transpose() ,
                                                                           d_sigmoid(np.dot(self.weights[i],
                                                                                            self.layers[i - 1]) +
                                                                                     self.biases[i]))))'''
                self.gradient_temp[i] = (
                    np.dot(self.weights[i+1].transpose(),self.gradient_temp[i+1])*
                    d_sigmoid(np.dot(self.weights[i],self.layers[i-1])+self.biases[i])
                                               )

                self.gradient_for_weight[i] = np.dot(self.gradient_temp[i].reshape(-1,1), self.layers[i - 1].reshape(1,-1))

            self.weights[i] -= learning_rate * self.gradient_for_weight[i]
            self.biases[i] -= learning_rate * self.gradient_temp[i]

            pass

        pass

if __name__ == '__main__':
    # 示例用法
    train_images = read_images('data\\train-images.idx3-ubyte')
    train_labels = read_labels('data\\train-labels.idx1-ubyte')
    network = Network(depth=3, layer_size=(16, 14, 10))

    test_images = read_images('data\\t10k-images.idx3-ubyte')
    test_labels = read_labels('data\\t10k-labels.idx1-ubyte')

    for i in range(60000):
        one_hot = np.zeros(10)
        one_hot[train_labels[i]] = 1
        temp=[]
        for j in range(28):
            temp += list(train_images[i][j])

        if 1:
            print(network.forward(np.array(temp)))
            print(network.calculate_cost(np.array(one_hot)))
        else:
            network.forward(np.array(temp))

        network.backward(np.array(one_hot),0.0001)

    count = 0
    for i in range(10000):
        one_hot = np.zeros(10)
        one_hot[test_labels[i]] = 1
        temp=[]
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

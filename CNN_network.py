from Automatic_differentiation import Tensor, read_labels, read_images, FCNN
import numpy as np

"""
使用知乎这篇文章的结构
https://zhuanlan.zhihu.com/p/227774699
"""


class CNN:
    def __init__(self):
        # 卷积核大小，池化大小

        self.kernel_size = (5, 3)
        self.pooling_size = (2, 2)

        # 输入的图片
        self.input = None

        # 卷积层
        self.conv_layers = None

        # 全连接层，调用FCNN类
        self.fc = FCNN(1, (10,))
        self.fcnn_input = None

        # 保存卷积层到全连接层的形状，更好地连接卷积层以及全连接层
        self.shape = None

        self.kernel = []

        for i in self.kernel_size:
            # 初始化卷积核
            # 这里使用np中的随机数生成，大小为i*i，范围为low ~ high
            self.kernel.append(Tensor(np.random.uniform(low=-1, high=1, size=(i, i))))

        self.convolutional_layer = []

    def forward(self, x: np.array):
        """
        前向传播
        :param x: 输入的图片，大小为28*28，类型为np.array
        :return:
        """
        # 初始化输入
        self.input = Tensor(x, False)

        # 卷积和池化操作
        # 当然，顺序和次数也可以魔改

        # 下面的代码看上去有点屎山，不要介意哈
        self.conv_layers = []
        # 卷积操作
        self.conv_layers.append(self.input.convolution_forward(self.kernel[0], 1))
        # 池化操作
        self.conv_layers.append(self.conv_layers[0].max_pooling(self.pooling_size))

        # 同上
        self.conv_layers.append(self.conv_layers[1].convolution_forward(self.kernel[1], 1))
        self.conv_layers.append(self.conv_layers[2].max_pooling(self.pooling_size))

        # 连接全连接层
        self.shape = self.conv_layers[3].shape

        # fcnn_input是对卷积层最后一层的引用
        self.fcnn_input = self.conv_layers[3]
        # 更改形状以适配全连接层 注意：反向传播时要改回形状，同时也要改回梯度的形状
        self.fcnn_input.data = self.fcnn_input.data.reshape(-1, 1)

        # 全连接层forward
        self.fc.forward(self.fcnn_input)

    def backward(self, y: np.array, learning_rate: float = 0.001):
        """
        反向传播
        :param learning_rate: 学习率，默认为0.001
        :param y: 标签，大小为10，类型为np.array，one-hot编码
        :return:
        """
        # 全连接层反向
        self.fc.backward(y, learning_rate)

        # 更改形状以适配卷积层
        self.fcnn_input.data = self.fcnn_input.data.reshape(self.shape)
        self.fcnn_input.grad = self.fcnn_input.grad.reshape(self.shape)

        # 卷积层反向
        self.conv_layers[3].convolution_backward(self.fcnn_input, self.kernel[1], learning_rate)
        self.conv_layers[2].max_pooling_backward(self.conv_layers[3], self.pooling_size)
        self.conv_layers[1].convolution_backward(self.conv_layers[2], self.kernel[0], learning_rate)
        self.conv_layers[0].max_pooling_backward(self.conv_layers[1], self.pooling_size)

        # 卷积核更新
        for i in range(len(self.kernel)):
            self.kernel[i].data -= learning_rate * self.kernel[i].grad
            self.kernel[i].grad = None

        # 清空梯度
        self.input.grad = None
        for i in self.conv_layers:
            i.grad = None



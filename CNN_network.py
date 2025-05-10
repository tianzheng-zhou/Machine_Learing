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

        # 全连接层，调用FCNN类
        self.fc = FCNN(1, (10,))

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
        for i in range(len(self.kernel)):
            # 卷积操作
            temp_conv =





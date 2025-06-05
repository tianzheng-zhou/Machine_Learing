# 导入必要的库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets

# 设置超参数
learning_rate = 1e-3  # 学习率
batch_size = 64  # 每个batch的大小
epochs = 5  # 训练轮数
loss_fn = nn.CrossEntropyLoss()  # 损失函数

# 尝试使用cuda等加速器加速，否则使用cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# 示例：使用fashionMNIST数据库
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 使用dataloader加载器
# batch_size越大，占用内存越大
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# 先定义神经网络类
class FCNN(nn.Module):
    def __init__(self):
        # 继承nn.Module
        super().__init__()

        # 将输入数据展平，即将二维张量转化为一维
        self.flatten = nn.Flatten()

        # 定义神经网络的结构
        # Sequential将多种运算聚合到一起
        self.linear_relu_stack = nn.Sequential(

            nn.Linear(28 * 28, 512),  # 线性层
            nn.ReLU(),  # 激活函数RELU

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        要使用模型，我们将输入数据传递给它。
        例如：
        模型实例为model
        X为输入特征
        则应当调用model(X)
        而不是使用model.forward()!
        这将执行模型的 forward 方法，以及一些后台操作。不要直接调用 model.forward()！

        小声bb：python还是有点不安全了，没有private和public的说法
        """

        # 先将输入数据展平
        x = self.flatten(x)

        # 接下来进行一次神经网络的前向运算
        # logits即为网络的原始输出
        logits = self.linear_relu_stack(x)

        # 然后返回原始输出
        return logits


# 然后定义训练循环以及测试循环
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # 将模型设置为训练模式——这对于批归一化和丢弃层至关重要
    # 在这种情况下并非必要，但为了最佳实践而添加
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 计算预测值
        pred = model(X)

        # 计算损失
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()

        # 每一百个批次打印进度，实时监控训练过程
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # 将模型设置为评估模式——这对于批归一化和Dropout层非常重要
    # 在此场景下虽非必需，但遵循最佳实践建议添加
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 创建模型实例
# 注意：要将模型实例放在选定的设备上（cpu或者gpu）
model = FCNN().to(device)

# 优化器设置（使用SGD优化器）
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 打印模型结构
print(model)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

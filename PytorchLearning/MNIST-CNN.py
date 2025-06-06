# 导入必要的库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets

# 设置超参数
learning_rate = 3e-4  # 学习率
batch_size = 10  # 每个batch的大小
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
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        self.linear_layers = nn.Sequential(
            nn.Linear(8 * 7 * 7, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        # 先通过卷积层
        x = self.conv_layers(x)
        # 然后展平
        x = self.flatten(x)
        # 最后通过全连接层
        logits = self.linear_layers(x)
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
model = CNN().to(device)

# 优化器设置（使用SGD优化器）
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 打印模型结构
print(model)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

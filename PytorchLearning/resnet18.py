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
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        # 先通过卷积层
        x = self.block(x)
        # 然后shortcut
        x = x + self.shortcut(x)
        # 最后通过relu
        logits = self.ReLU(x)
        return logits

class Resnet18(nn.Module):
    def __init__(self, res_block, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res_layers = nn.Sequential(
            res_block(64, 64, 1),
            res_block(64, 64, 1),
            res_block(64, 128, 2),
            res_block(128, 128, 1),
        )

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

# 优化器设置（使用Adam优化器）
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 打印模型结构
print(model)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

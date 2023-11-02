import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义模型
class SimpleCNN(nn.Module):
  def __init__(self):
      super(SimpleCNN, self).__init__()
      self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
      self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
      self.fc1 = nn.Linear(320, 50)
      self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
      x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
      x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
      x = x.view(-1, 320)
      x = nn.functional.relu(self.fc1(x))
      x = self.fc2(x)
      return nn.functional.log_softmax(x, dim=1)


  
if __name__ == '__main__':
    # 获取 cpu, gpu 或 mps 设备用于加速训练.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # 定义数据预处理
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # 定义数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    model = SimpleCNN()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # 进行多轮训练
    num_epochs = 10
    for epoch in range(num_epochs):
    # 进行训练
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(images), len(train_loader.dataset),
                    100. * i / len(train_loader), loss.item()))

    # 进行测试
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss += criterion(output, labels).item()  # 将一批的损失相加
            pred = output.argmax(dim=1, keepdim=True)  # 找到概率最大的下标
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    pass
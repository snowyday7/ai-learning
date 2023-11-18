# 四 MLP 多层感知机

## 1 MLP

MLP（多层感知机，Multi-Layer Perceptron）是一种人工智能技术，它是一种神经网络模型，主要用于解决非线性问题。MLP 包括一个或多个隐层，除了输入层、隐层，还有输出层。隐层和输出层的神经元通过线性和非线性组合，对输入数据进行处理，以达到预定的目标，如图像识别、语音识别等。
MLP 的激活函数和损失函数可以选择不同的类型，例如，激活函数可以选择 ReLU、tanh 等，损失函数可以选择均方误差（MSE）等。

- sklearn


```python
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score

# 创建数据集  
X = np.random.rand(100, 10)  # 100 个样本，每个样本 10 维  
y = np.random.randint(0, 2, (100, 1))  # 100 个样本，二分类问题

# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 MLP 分类器  
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)

# 训练模型  
mlp.fit(X_train, y_train)

# 预测  
y_pred = mlp.predict(X_test)

# 计算准确率  
accuracy = accuracy_score(y_test, y_pred)  
print('Accuracy:', accuracy)  

```


```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(15,15), random_state=1)

clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
```

- pytorch


```python
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
from torchvision import datasets, transforms

# 加载 MNIST 数据集  
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)  
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  
testloader = DataLoader(testset, batch_size=100, shuffle=False)

# 创建 MLP 分类器模型  
class MLP(nn.Module):  
    def __init__(self):  
        super(MLP, self).__init__()  
        self.fc1 = nn.Linear(28 * 28, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):  
        x = x.view(-1, 28 * 28)  
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x

# 初始化模型、损失函数和优化器  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
model = MLP().to(device)  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型  
for epoch in range(10):  
    running_loss = 0.0  
    for i, data in enumerate(trainloader, 0):  
        inputs, labels = data  
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()

        running_loss += loss.item()  
    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / (i + 1)))

model.eval()  
with torch.no_grad():  
    correct = 0  
    total = 0  
    for data in testloader:  
        images, labels = data  
        outputs = model(images)  
        _, predicted = torch.max(outputs.data, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')  

```

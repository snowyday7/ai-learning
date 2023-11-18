# 二 LogistRegression 逻辑回归

## 1 逻辑回归

给定数据$X=\{x_1,x_2,...,\}$,$Y=\{y_1,y_2,...,\}$
考虑二分类任务，即$y_i\in{\{0,1\}},i=1,2,...$,

- 公式：

$$
h_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} x}}
$$
        或
$$
h_{\theta}(x)=g\left(\theta^{T} x\right), g(z)=\frac{1}{1+e^{-z}}
$$

- 损失函数：

$$
J(\theta)=-\frac{1}{m}\left[\sum_{i=1}^{m}\left(y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]\right.
$$

- 梯度：

$$
\frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m} \sum_{i=0}^{m}\left(h_{\theta}-y^{i}\left(x^{i}\right)\right) x_{j}^{i}
$$

- sklearn


```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

# 数据
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

clf = LogisticRegression(penalty="l1", solver="saga", tol=0.1)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Test score with L1 penalty: %.4f" % score)
```

- numpy


```python
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
p_parent_path = str(Path().absolute().parent.parent)
sys.path.append(p_parent_path) 
print(f"主目录为：{p_parent_path}")

train_dataset = datasets.MNIST(root = p_parent_path+'/datasets/', train = True, transform = transforms.ToTensor(), download = False)
test_dataset = datasets.MNIST(root = p_parent_path+'/datasets/', train = False, transform = transforms.ToTensor(), download = False)

batch_size = len(train_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
X_train,y_train = next(iter(train_loader))
X_test,y_test = next(iter(test_loader))
# 打印前100张图片
images, labels= X_train[:100], y_train[:100] 
# 使用images生成宽度为10张图的网格大小
img = torchvision.utils.make_grid(images, nrow=10)
# cv2.imshow()的格式是(size1,size1,channels),而img的格式是(channels,size1,size1),
# 所以需要使用.transpose()转换，将颜色通道数放至第三维
img = img.numpy().transpose(1,2,0)
print(images.shape)
print(labels.reshape(10,10))
print(img.shape)
plt.imshow(img)
plt.show()

X_train,y_train = X_train.cpu().numpy(),y_train.cpu().numpy() # tensor转为array形式)
X_test,y_test = X_test.cpu().numpy(),y_test.cpu().numpy() # tensor转为array形式)
print(f"数据格式：{type(X_train)}，数据维度：{X_train.shape}")
print(f"数据格式：{type(y_train)}，数据维度：{y_train.shape}")
X_train = X_train.reshape(X_train.shape[0],784)
print(f"数据格式：{type(X_train)}，数据维度：{X_train.shape}")

ones_col=[[1] for i in range(len(X_train))] # 生成全为1的二维嵌套列表，即[[1],[1],...,[1]]
X_train_modified=np.append(X_train,ones_col,axis=1)
x_train_modified_mat = np.mat(X_train_modified)
# Mnsit有0-9十个标记，由于是二分类任务，所以可以将标记0的作为1，其余为0用于识别是否为0的任务
y_train_modified=np.array([1 if y_train[i]==1 else 0 for i in range(len(y_train))])

theta = np.mat(np.zeros(len(X_train_modified[0])))
n_epochs=10 
lr = 0.01 # 学习率

def sigmoid(x):
    '''sigmoid函数
    '''
    return 1.0/(1+np.exp(-x))

train_dataset = datasets.MNIST(p_parent_path+'/datasets/', train=True, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
images, labels = next(iter(train_loader))
print(images)
# 使用images生成宽度为10张图的网格大小
img = torchvision.utils.make_grid(images, nrow=10)
# cv2.imshow()的格式是(size1,size1,channels),而img的格式是(channels,size1,size1),
# 所以需要使用.transpose()转换，将颜色通道数放至第三维
img = img.numpy().transpose(1,2,0)
# print(images.shape)
# print(labels.reshape(10,10))
plt.imshow(img)
plt.show()


for i_epoch in range(n_epochs):
    loss_epoch = 0
    for i in range(len(X_train_modified)):
        hypothesis = sigmoid(np.dot(X_train_modified[i], theta.T))
        error = y_train_modified[i]- hypothesis
        grad = error*x_train_modified_mat[i]
        theta += lr*grad
        loss_epoch+=error.item()
    # loss_epoch /= len(X_train_modified)
    print(f"回合数：{i_epoch+1}/{n_epochs}，损失：{loss_epoch:.4f}")
```

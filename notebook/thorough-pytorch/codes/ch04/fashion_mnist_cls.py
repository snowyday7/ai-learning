import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
from torchvision import transforms

# ## 读取方式一：使用torchvision自带数据集，下载可能需要一段时间
# from torchvision import datasets

# train_data = datasets.FashionMNIST(root='./', train=True, download=True, transform=data_transform)
# test_data = datasets.FashionMNIST(root='./', train=False, download=True, transform=data_transform)

## 读取方式二：读入csv格式的数据，自行构建Dataset类
# csv数据下载链接：https://www.kaggle.com/zalando-research/fashionmnist
class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# 加载数据
def load_fm_data(data_transform, batch_size, num_workers):
    current_dir = sys.path[0]  # 当前脚本目录
    parent_dir = os.path.dirname(current_dir)  # 上层目录
    parent_dir = os.path.dirname(parent_dir)  # 上层目录
    data_dir = os.path.join(parent_dir, 'datasets/ch04') # data目录

    train_csv_file = os.path.join(data_dir, 'fashion_mnist_train.csv')
    test_csv_file = os.path.join(data_dir, 'fashion_mnist_test.csv')
    # train_csv_file = "../datasets/fashion_mnist_train.csv"
    # test_csv_file = "../datasets/fashion_mnist_test.csv"

    train_df = pd.read_csv(train_csv_file)
    test_df = pd.read_csv(test_csv_file)
    train_data = FMDataset(train_df, data_transform)
    test_data = FMDataset(test_df, data_transform)

    # 在构建训练和测试数据集完成后，需要定义DataLoader类，以便在训练和测试时加载数据
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 读入后，我们可以做一些数据可视化操作，主要是验证我们读入的数据是否正确
    image, label = next(iter(train_loader))
    print(image.shape, label.shape)
    plt.imshow(image[0][0], cmap="gray")
    plt.show()

    return train_loader, test_loader

# 使用CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

def train(epoch, model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for i, (data, label) in enumerate(train_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        if i % 100 == 0:
            print('--Epoch: {} \tTraining Loss: {:.6f} [{}/{} ({:.0f}%)]'.format(epoch, loss.item(), i * len(data), len(train_loader.dataset),
              100. * i / len(train_loader)))
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

def val(epoch, model, device, test_loader, criterion):       
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            # data, label = data.cuda(), label.cuda()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data, label = data.to(device), label.to(device)
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))



if __name__ == '__main__':
    # 配置GPU，这里有两种方式
    ## 方案一：使用os.environ
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    ## 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
    batch_size = 256
    num_workers = 4   # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    lr = 1e-4
    epochs = 20

    # 首先设置数据变换
    image_size = 28
    data_transform = transforms.Compose([
        transforms.ToPILImage(),  
        # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_loader, test_loader = load_fm_data(data_transform, batch_size, num_workers)

    model = Net()
    # model = model.cuda()  #将模型放于GPU上
    # model = nn.DataParallel(model).cuda()   # 多卡训练时的写法

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=[1,1,1,1,3,1,1,1,1,1])

    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # 训练及验证
    for epoch in range(1, epochs+1):
        train(epoch, model, device, train_loader, optimizer, criterion)
        val(epoch, model, device, test_loader, criterion)
    
    # save_path = "./FahionModel.pkl"
    # torch.save(model, save_path)
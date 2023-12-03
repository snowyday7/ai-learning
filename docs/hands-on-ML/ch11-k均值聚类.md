# 十一 k均值聚类

## 1 自定义实现Kmeans


```python
import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('./data/ch11/kmeans_data.csv', delimiter=',')
print('数据集大小：', len(dataset))
```

    数据集大小： 80



```python
# 绘图函数
def show_cluster(dataset, cluster, centroids=None):  
    # dataset：数据
    # centroids：聚类中心点的坐标
    # cluster：每个样本所属聚类
    # 不同种类的颜色，用以区分划分的数据的类别
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', '^', 's', 'd']
    # 画出所有样例
    K = len(np.unique(cluster))
    for i in range(K):
        plt.scatter(dataset[cluster == i, 0], dataset[cluster == i, 1], 
            color=colors[i], marker=markers[i])

    # 画出中心点
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], 
            color=colors[:K], marker='+', s=150)  
        
    plt.show()

# 初始时不区分类别
show_cluster(dataset, np.zeros(len(dataset), dtype=int))
```


    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_3_0.png)
    



```python
def random_init(dataset, K):
    # 随机选取是不重复的
    idx = np.random.choice(np.arange(len(dataset)), size=K, replace=False)
    return dataset[idx]
```


```python
def Kmeans(dataset, K, init_cent):
    # dataset：数据集
    # K：目标聚类数
    # init_cent：初始化中心点的函数
    centroids = init_cent(dataset, K)
    cluster = np.zeros(len(dataset), dtype=int)
    changed = True
    # 开始迭代
    itr = 0
    while changed:
        changed = False
        loss = 0
        for i, data in enumerate(dataset):
            # 寻找最近的中心点
            dis = np.sum((centroids - data) ** 2, axis=-1)
            k = np.argmin(dis)
            # 更新当前样本所属的聚类
            if cluster[i] != k:
                cluster[i] = k
                changed = True
            # 计算损失函数
            loss += np.sum((data - centroids[k]) ** 2)
        # 绘图
        print(f'Iteration {itr}, Loss {loss:.3f}')
        show_cluster(dataset, cluster, centroids)
        # 更新中心点
        for i in range(K):
            centroids[i] = np.mean(dataset[cluster == i], axis=0)
        itr += 1

    return centroids, cluster
```


```python
np.random.seed(0)
cent, cluster = Kmeans(dataset, 4, random_init)
```

    Iteration 0, Loss 711.336



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_6_1.png)
    


    Iteration 1, Loss 409.495



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_6_3.png)
    


    Iteration 2, Loss 395.264



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_6_5.png)
    


    Iteration 3, Loss 346.068



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_6_7.png)
    


    Iteration 4, Loss 294.244



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_6_9.png)
    


    Iteration 5, Loss 178.808



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_6_11.png)
    


    Iteration 6, Loss 151.090



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_6_13.png)
    



```python
def kmeanspp_init(dataset, K):
    # 随机第一个中心点
    idx = np.random.choice(np.arange(len(dataset)))
    centroids = dataset[idx][None]
    for k in range(1, K):
        d = []
        # 计算每个点到当前中心点的距离
        for data in dataset:
            dis = np.sum((centroids - data) ** 2, axis=-1)
            # 取最短距离的平方
            d.append(np.min(dis) ** 2)
        # 归一化
        d = np.array(d)
        d /= np.sum(d)
        # 按概率选取下一个中心点
        cent_id = np.random.choice(np.arange(len(dataset)), p=d)
        cent = dataset[cent_id]
        centroids = np.concatenate([centroids, cent[None]], axis=0)

    return centroids
```


```python
cent, cluster = Kmeans(dataset, 4, kmeanspp_init)
```

    Iteration 0, Loss 373.939



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_8_1.png)
    


    Iteration 1, Loss 158.147



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_8_3.png)
    


    Iteration 2, Loss 151.273



    
![png](ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_files/ch11-k%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB_8_5.png)
    


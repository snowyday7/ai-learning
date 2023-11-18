# 十 kNN k最近邻

# 1 kNN


k最近邻（K Nearest Neighbors），监督学习算法，用于分类和回归。如果一个样本在特征空间中的K个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别

- k值的选取

    一般选择一个较小的数值，通常采用交叉验证的方法来选择最优的 K 值

- 交叉验证方法
  - Hold-out：随机从最初的样本中选出部分，形成交叉验证数据，而剩余的就当做训练数据
  - K折交叉验证：将数据集分成k份，将其中一份作为测试集，将剩余的作为训练集，重复k次。平均K次的结果或者使用其它结合方式，最终得到一个单一估测
  - 留一法：将数据集分成k份，将其中一份作为测试集，将剩余的作为训练集，重复k次。这个步骤一直持续到每个样本都被当做一次验证资料

- 分类决策
  - 多数表决法：少数服从多数
  - 加权表决法：在数据之间有权重的情况下，一般采用加权表决法

- 算法步骤：
  1. 准备数据：获取带有标签的训练数据集，包括输入特征和对应的类别或数值
  2. 计算距离：对于给定的测试样本，计算其与训练数据集中各个样本之间的距离。常用的距离度量方法有欧氏距离和曼哈顿距离等
  3. 选择k值：选择一个合适的k值，即要考虑的最近邻样本的数量。这个值可以根据数据集的大小和问题的特点来确定
  4. 选取最近邻：从距离计算中选择k个最近邻样本
  5. 判定结果：对于分类问题，通过多数表决决定测试样本的类别，即选择k个最近邻中出现最多次数的类别作为预测结果。对于回归问题，可以取k个最近邻的平均值作为预测结果

- sklearn


```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#加载数据集
digits = load_digits()
data = digits.data     # 特征集
target = digits.target # 目标集

#将数据集拆分为训练集（75%）和测试集（25%）:
train_x, test_x, train_y, test_y = train_test_split(
    data, target, test_size=0.25, random_state=33)

#构造KNN分类器：采用默认参数
knn = KNeighborsClassifier() 

#拟合模型：
knn.fit(train_x, train_y) 
#预测数据：
predict_y = knn.predict(test_x) 

#计算模型准确度
score = accuracy_score(test_y, predict_y)
print(score)
```

    0.9844444444444445


- 自定义实现


```python
class KNN:
    def __init__(self, x_train, y_train, x_test, y_test, k):
        '''
        Args:
            x_train [Array]: 训练集数据
            y_train [Array]: 训练集标签
            x_test [Array]: 测试集数据
            y_test [Array]: 测试集标签
            k [int]: k of kNN
        '''
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        # 将输入数据转为矩阵形式，方便运算
        self.x_train_mat, self.x_test_mat = np.mat(
            self.x_train), np.mat(self.x_test)
        self.y_train_mat, self.y_test_mat = np.mat(
            self.y_test).T, np.mat(self.y_test).T
        self.k = k

    def _calc_dist(self, x1, x2):
        '''计算两个样本点向量之间的距离,使用的是欧氏距离
        :param x1:向量1
        :param x2:向量2
        :return: 向量之间的欧式距离
        '''
        return np.sqrt(np.sum(np.square(x1 - x2)))

    def _get_k_nearest(self,x):
        '''
        预测样本x的标记。
        获取方式通过找到与样本x最近的topK个点，并查看它们的标签。
        查找里面占某类标签最多的那类标签
        :param trainDataMat:训练集数据集
        :param trainLabelMat:训练集标签集
        :param x:待预测的样本x
        :param topK:选择参考最邻近样本的数目（样本数目的选择关系到正确率，详看3.2.3 K值的选择）
        :return:预测的标记
        '''
        # 初始化距离列表，dist_list[i]表示待预测样本x与训练集中第i个样本的距离
        dist_list=[0]* len(self.x_train_mat)

        # 遍历训练集中所有的样本点，计算与x的距离
        for i in range( len(self.x_train_mat)):
            # 获取训练集中当前样本的向量
            x0 = self.x_train_mat[i]
            # 计算向量x与训练集样本x0的距离
            dist_list[i] = self._calc_dist(x0, x)

        # 对距离列表排序并返回距离最近的k个训练样本的下标
        # ----------------优化点-------------------
        # 由于我们只取topK小的元素索引值，所以其实不需要对整个列表进行排序，而argsort是对整个
        # 列表进行排序的，存在时间上的浪费。字典有现成的方法可以只排序top大或top小，可以自行查阅
        # 对代码进行稍稍修改即可
        # 这里没有对其进行优化主要原因是KNN的时间耗费大头在计算向量与向量之间的距离上，由于向量高维
        # 所以计算时间需要很长，所以如果要提升时间，在这里优化的意义不大。
        k_nearest_index = np.argsort(np.array(dist_list))[:self.k]  # 升序排序
        return k_nearest_index


    def _predict_y(self,k_nearest_index):
        # label_list[1]=3，表示label为1的样本数有3个，由于此处label为0-9，可以初始化长度为10的label_list
        label_list=[0] * 10
        for index in k_nearest_index:
            one_hot_label=self.y_train[index]
            number_label=np.argmax(one_hot_label)
            label_list[number_label] += 1
        # 采用投票法，即样本数最多的label就是预测的label
        y_predict=label_list.index(max(label_list))
        return y_predict

    def test(self,n_test=200):
        '''
        测试正确率
        :param: n_test: 待测试的样本数
        :return: 正确率
        '''
        print('start test')

        # 错误值计数
        error_count = 0
        # 遍历测试集，对每个测试集样本进行测试
        # 由于计算向量与向量之间的时间耗费太大，测试集有6000个样本，所以这里人为改成了
        # 测试200个样本点，若要全跑，更改n_test即可
        for i in range(n_test):
            # print('test %d:%d'%(i, len(trainDataArr)))
            print('test %d:%d' % (i, n_test))
            # 读取测试集当前测试样本的向量
            x = self.x_test_mat[i]
            # 获取距离最近的训练样本序号
            k_nearest_index=self._get_k_nearest(x)
            # 预测输出y
            y=self._predict_y(k_nearest_index)
            # 如果预测label与实际label不符，错误值计数加1
            if y != np.argmax(self.y_test[i]):
                error_count += 1
            print("accuracy=",1 - (error_count /(i+1)))

        # 返回正确率
        return 1 - (error_count / n_test)
```

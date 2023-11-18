# 八 AdaBoost

## 1 AdaBoost

boosting是一族将弱学习器提升为强学习器的算法。这族算法的工作机制是：先从初试训练集训练出一个基学习器，再根据基学习器的表现对训练样本进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基分类器，如此重复进行，直至基学习器数目达到实现指定值T，最后将这T个基学习器进行加权组合

AdaBoost算法有多种推导方式，比较容易理解的是基于“加性模型”，即基学习器的线性组合

- 公式

$$
H(x)=\displaystyle\sum_{t=1}^T \alpha_ih_t(x)
$$

其中$h_i(x),i=1,2,...$表示基分类器，$\alpha_i$是每个基分类器对应的权重，表示如下：

$$
\alpha_{i}=\frac{1}{2} \ln \left(\frac{1-\epsilon_{i}}{\epsilon_{i}}\right)
$$

其中$\epsilon_{i}$是每个弱分类器的错误率。

- 损失函数

$$
l_{exp}(H|D)=E_{X D}[e^{-f(x)H(x)}]
$$

- sklearn


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

wine = load_wine()#使用葡萄酒数据集
print(f"所有特征：{wine.feature_names}")
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print(f"训练数据量：{len(X_train)}，测试数据量：{len(X_test)}")

#构建并训练决策树分类器，这里特征选择标准使用基尼指数，树的最大深度为1
base_model = DecisionTreeClassifier(max_depth=1, criterion='gini',random_state=1).fit(X_train, y_train)
y_pred = base_model.predict(X_test)#对训练集进行预测
print(f"决策树的准确率：{accuracy_score(y_test,y_pred):.3f}")

# 定义模型，这里最大分类器数量为50，学习率为1.5
model = AdaBoostClassifier(estimator=base_model,n_estimators=50,learning_rate=0.8)
# 训练
model.fit(X_train, y_train) 
# 预测
y_pred = model.predict(X_test) 
acc = metrics.accuracy_score(y_test, y_pred) # 准确率
print(f"准确率：{acc:.2}")
```

    所有特征：['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    训练数据量：142，测试数据量：36
    决策树的准确率：0.694
    准确率：0.94



```python

hyperparameter_space = {'n_estimators':list(range(2, 102, 2)), 
                        'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
# 使用准确率为标准，将得到的准确率最高的参数输出，cv=5表示交叉验证参数，这里使用五折交叉验证，n_jobs=-1表示并行数和cpu一致
gs = GridSearchCV(AdaBoostClassifier(
                                     algorithm='SAMME.R',
                                     random_state=1),
                  param_grid=hyperparameter_space, 
                  scoring="accuracy", n_jobs=-1, cv=5)

gs.fit(X_train, y_train)
print("最优超参数:", gs.best_params_)
print("最优模型:", gs.best_estimator_)
print("最优分数:", gs.best_score_)
```

    最优超参数: {'learning_rate': 0.8, 'n_estimators': 24}
    最优模型: AdaBoostClassifier(learning_rate=0.8, n_estimators=24, random_state=1)
    最优分数: 0.9857142857142858


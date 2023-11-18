# 三 DecisionTree 决策树

## 1 决策树

决策树模型就是数据结构中的树，根据**特征选择依据**(信息熵)等划分特征，生成决策树，然后**剪枝**提高泛化能力，可分类可回归，代表算法有ID3，C4.5和CART

- sklearn


```python
import seaborn as sns
from pandas import plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree

# 加载数据集
data = load_iris() 
# 转换成.DataFrame形式
df = pd.DataFrame(data.data, columns = data.feature_names)
# 添加品种列
df['Species'] = data.target

# 用数值替代品种名作为标签
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

# 提取数据和标签
X = df.drop(columns="Species")
y = df["Species"]
feature_names = X.columns
labels = y.unique()

X_train, test_x, y_train, test_lab = train_test_split(X,y,
                                                 test_size = 0.4,
                                                 random_state = 42)
model = DecisionTreeClassifier(max_depth =3, random_state = 42)
model.fit(X_train, y_train) 
# 以文字形式输出树     
text_representation = tree.export_text(model)
print(text_representation)
# 用图片画出
plt.figure(figsize=(30,10), facecolor ='g') #
a = tree.plot_tree(model,
                   feature_names = feature_names,
                   class_names = labels,
                   rounded = True,
                   filled = True,
                   fontsize=14)
plt.show()                                          
```

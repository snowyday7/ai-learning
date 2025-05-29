# Numpy 从基础到进阶的详细使用案例


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
```

## 1. Numpy基础操作


```python
print("="*50)
print("1. Numpy基础操作")
print("="*50)

# 创建数组的多种方式
print("\n1.1 创建数组的多种方式")
array1 = np.array([1, 2, 3, 4, 5])  # 从列表创建
array2 = np.array([[1, 2, 3], [4, 5, 6]])  # 二维数组
array3 = np.zeros((3, 4))  # 全零数组
array4 = np.ones((2, 3))   # 全一数组
array5 = np.full((2, 2), 7)  # 填充指定值
array6 = np.eye(3)  # 单位矩阵
array7 = np.arange(0, 10, 2)  # 等差数列
array8 = np.linspace(0, 1, 5)  # 线性空间
array9 = np.random.random((2, 3))  # 随机数组

print(f"一维数组: {array1}")
print(f"二维数组:\n{array2}")
print(f"全零数组:\n{array3}")
print(f"全一数组:\n{array4}")
print(f"填充数组:\n{array5}")
print(f"单位矩阵:\n{array6}")
print(f"等差数列: {array7}")
print(f"线性空间: {array8}")
print(f"随机数组:\n{array9}")
```

    ==================================================
    1. Numpy基础操作
    ==================================================
    
    1.1 创建数组的多种方式
    一维数组: [1 2 3 4 5]
    二维数组:
    [[1 2 3]
     [4 5 6]]
    全零数组:
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    全一数组:
    [[1. 1. 1.]
     [1. 1. 1.]]
    填充数组:
    [[7 7]
     [7 7]]
    单位矩阵:
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    等差数列: [0 2 4 6 8]
    线性空间: [0.   0.25 0.5  0.75 1.  ]
    随机数组:
    [[0.37148783 0.14593383 0.08151387]
     [0.88250674 0.11092649 0.44299836]]


### 1.2 数组属性


```python
test_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"数组形状: {test_array.shape}")
print(f"数组维度: {test_array.ndim}")
print(f"数组大小: {test_array.size}")
print(f"数据类型: {test_array.dtype}")
print(f"每个元素字节数: {test_array.itemsize}")
print(f"总字节数: {test_array.nbytes}")
```

    数组形状: (2, 2, 2)
    数组维度: 3
    数组大小: 8
    数据类型: int64
    每个元素字节数: 8
    总字节数: 64


## 2. 数组形状操作


```python
print("\n" + "="*50)
print("2. 数组形状操作")
print("="*50)

original = np.arange(12)
print(f"原始数组: {original}")

# reshape - 改变形状
reshaped = original.reshape(3, 4)
print(f"重塑为3x4:\n{reshaped}")

# resize - 原地改变大小
resized = original.copy()
resized.resize(2, 6)
print(f"调整大小为2x6:\n{resized}")

# flatten vs ravel
flattened = reshaped.flatten()  # 返回副本
raveled = reshaped.ravel()      # 返回视图
print(f"展平数组: {flattened}")
print(f"拉平数组: {raveled}")

# 转置
transposed = reshaped.T
print(f"转置数组:\n{transposed}")

# swapaxes - 交换轴
swapped = reshaped.swapaxes(0, 1)
print(f"交换轴:\n{swapped}")
```

    
    ==================================================
    2. 数组形状操作
    ==================================================
    原始数组: [ 0  1  2  3  4  5  6  7  8  9 10 11]
    重塑为3x4:
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    调整大小为2x6:
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]]
    展平数组: [ 0  1  2  3  4  5  6  7  8  9 10 11]
    拉平数组: [ 0  1  2  3  4  5  6  7  8  9 10 11]
    转置数组:
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]
    交换轴:
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]


## 3. 数组索引和切片


```python
print("\n" + "="*50)
print("3. 数组索引和切片")
print("="*50)

arr = np.arange(24).reshape(4, 6)
print(f"原始数组:\n{arr}")

# 基本索引
print(f"\n3.1 基本索引")
print(f"第2行第3列: {arr[1, 2]}")
print(f"第1行: {arr[0]}")
print(f"第3列: {arr[:, 2]}")
print(f"前2行前3列:\n{arr[:2, :3]}")

# 高级索引
print(f"\n3.2 高级索引")
# 整数数组索引
rows = np.array([0, 2, 3])
cols = np.array([1, 3, 5])
print(f"指定位置元素: {arr[rows, cols]}")

# 布尔索引
mask = arr > 10
print(f"大于10的元素: {arr[mask]}")

# 花式索引
fancy_indexed = arr[[0, 2], :][:, [1, 3, 5]]
print(f"花式索引结果:\n{fancy_indexed}")
```

    
    ==================================================
    3. 数组索引和切片
    ==================================================
    原始数组:
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]]
    
    3.1 基本索引
    第2行第3列: 8
    第1行: [0 1 2 3 4 5]
    第3列: [ 2  8 14 20]
    前2行前3列:
    [[0 1 2]
     [6 7 8]]
    
    3.2 高级索引
    指定位置元素: [ 1 15 23]
    大于10的元素: [11 12 13 14 15 16 17 18 19 20 21 22 23]
    花式索引结果:
    [[ 1  3  5]
     [13 15 17]]


## 4. 数组运算


```python
print("\n" + "="*50)
print("4. 数组运算")
print("="*50)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(f"数组a:\n{a}")
print(f"数组b:\n{b}")

# 基本运算
print(f"\n4.1 基本运算")
print(f"加法:\n{a + b}")
print(f"减法:\n{a - b}")
print(f"乘法(元素级):\n{a * b}")
print(f"除法:\n{a / b}")
print(f"幂运算:\n{a ** 2}")

# 矩阵运算
print(f"\n4.2 矩阵运算")
print(f"矩阵乘法:\n{np.dot(a, b)}")
print(f"矩阵乘法(新语法):\n{a @ b}")

# 通用函数
print(f"\n4.3 通用函数")
angles = np.array([0, np.pi/4, np.pi/2, np.pi])
print(f"角度: {angles}")
print(f"正弦值: {np.sin(angles)}")
print(f"余弦值: {np.cos(angles)}")
print(f"指数: {np.exp([1, 2, 3])}")
print(f"对数: {np.log([1, np.e, np.e**2])}")
```

    
    ==================================================
    4. 数组运算
    ==================================================
    数组a:
    [[1 2]
     [3 4]]
    数组b:
    [[5 6]
     [7 8]]
    
    4.1 基本运算
    加法:
    [[ 6  8]
     [10 12]]
    减法:
    [[-4 -4]
     [-4 -4]]
    乘法(元素级):
    [[ 5 12]
     [21 32]]
    除法:
    [[0.2        0.33333333]
     [0.42857143 0.5       ]]
    幂运算:
    [[ 1  4]
     [ 9 16]]
    
    4.2 矩阵运算
    矩阵乘法:
    [[19 22]
     [43 50]]
    矩阵乘法(新语法):
    [[19 22]
     [43 50]]
    
    4.3 通用函数
    角度: [0.         0.78539816 1.57079633 3.14159265]
    正弦值: [0.00000000e+00 7.07106781e-01 1.00000000e+00 1.22464680e-16]
    余弦值: [ 1.00000000e+00  7.07106781e-01  6.12323400e-17 -1.00000000e+00]
    指数: [ 2.71828183  7.3890561  20.08553692]
    对数: [0. 1. 2.]


## 5. 统计函数


```python
print("\n" + "="*50)
print("5. 统计函数")
print("="*50)

data = np.random.randn(4, 5) * 10 + 50  # 正态分布数据
print(f"随机数据:\n{data}")

print(f"\n5.1 基本统计")
print(f"均值: {np.mean(data):.2f}")
print(f"中位数: {np.median(data):.2f}")
print(f"标准差: {np.std(data):.2f}")
print(f"方差: {np.var(data):.2f}")
print(f"最小值: {np.min(data):.2f}")
print(f"最大值: {np.max(data):.2f}")
print(f"极差: {np.ptp(data):.2f}")

print(f"\n5.2 按轴统计")
print(f"按行均值: {np.mean(data, axis=1)}")
print(f"按列均值: {np.mean(data, axis=0)}")

print(f"\n5.3 百分位数")
percentiles = [25, 50, 75, 90, 95]
for p in percentiles:
    print(f"{p}%分位数: {np.percentile(data, p):.2f}")
```

    
    ==================================================
    5. 统计函数
    ==================================================
    随机数据:
    [[44.19792053 47.54031908 39.63217262 50.86050705 72.0721636 ]
     [27.19453101 58.1681018  57.41304455 52.42596706 60.09416829]
     [49.40828219 53.95984009 59.80751489 48.7735961  61.39712276]
     [46.21592246 41.68052865 32.75953276 49.93128526 43.05168684]]
    
    5.1 基本统计
    均值: 49.83
    中位数: 49.67
    标准差: 10.14
    方差: 102.84
    最小值: 27.19
    最大值: 72.07
    极差: 44.88
    
    5.2 按轴统计
    按行均值: [50.86061658 51.05916254 54.6692712  42.7277912 ]
    按列均值: [41.75416405 50.3371974  47.40306621 50.49783887 59.15378537]
    
    5.3 百分位数
    25%分位数: 43.91
    50%分位数: 49.67
    75%分位数: 57.60
    90%分位数: 60.22
    95%分位数: 61.93


## 6. 数组合并和分割


```python
print("\n" + "="*50)
print("6. 数组合并和分割")
print("="*50)

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(f"数组1:\n{arr1}")
print(f"数组2:\n{arr2}")

# 合并
print(f"\n6.1 数组合并")
print(f"垂直合并:\n{np.vstack((arr1, arr2))}")
print(f"水平合并:\n{np.hstack((arr1, arr2))}")
print(f"深度合并:\n{np.dstack((arr1, arr2))}")
print(f"concatenate合并(axis=0):\n{np.concatenate((arr1, arr2), axis=0)}")
print(f"concatenate合并(axis=1):\n{np.concatenate((arr1, arr2), axis=1)}")

# 分割
print(f"\n6.2 数组分割")
big_array = np.arange(16).reshape(4, 4)
print(f"原数组:\n{big_array}")

v_split = np.vsplit(big_array, 2)
print(f"垂直分割: {len(v_split)}个数组")
for i, arr in enumerate(v_split):
    print(f"第{i+1}个:\n{arr}")

h_split = np.hsplit(big_array, 2)
print(f"水平分割: {len(h_split)}个数组")
```

    
    ==================================================
    6. 数组合并和分割
    ==================================================
    数组1:
    [[1 2]
     [3 4]]
    数组2:
    [[5 6]
     [7 8]]
    
    6.1 数组合并
    垂直合并:
    [[1 2]
     [3 4]
     [5 6]
     [7 8]]
    水平合并:
    [[1 2 5 6]
     [3 4 7 8]]
    深度合并:
    [[[1 5]
      [2 6]]
    
     [[3 7]
      [4 8]]]
    concatenate合并(axis=0):
    [[1 2]
     [3 4]
     [5 6]
     [7 8]]
    concatenate合并(axis=1):
    [[1 2 5 6]
     [3 4 7 8]]
    
    6.2 数组分割
    原数组:
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    垂直分割: 2个数组
    第1个:
    [[0 1 2 3]
     [4 5 6 7]]
    第2个:
    [[ 8  9 10 11]
     [12 13 14 15]]
    水平分割: 2个数组


## 7. 线性代数


```python
print("\n" + "="*50)
print("7. 线性代数")
print("="*50)

# 创建矩阵
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
b_vec = np.array([1, 2, 3]) # Renamed to avoid conflict with previous 'b'

print(f"矩阵A:\n{A}")
print(f"向量b: {b_vec}")

print(f"\n7.1 基本线性代数运算")
print(f"矩阵行列式: {np.linalg.det(A):.2f}")
print(f"矩阵迹: {np.trace(A)}")
print(f"矩阵秩: {np.linalg.matrix_rank(A)}")

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\n7.2 特征值和特征向量")
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 奇异值分解
U, s, Vt = np.linalg.svd(A)
print(f"\n7.3 奇异值分解")
print(f"U矩阵形状: {U.shape}")
print(f"奇异值: {s}")
print(f"Vt矩阵形状: {Vt.shape}")

# 求解线性方程组
try:
    x_sol = np.linalg.solve(A, b_vec) # Renamed to avoid conflict
    print(f"\n7.4 线性方程组解: {x_sol}")
    print(f"验证Ax: {A @ x_sol}")
except np.linalg.LinAlgError:
    print("\n7.4 矩阵奇异，使用最小二乘解")
    x_sol = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    print(f"最小二乘解: {x_sol}")
```

    
    ==================================================
    7. 线性代数
    ==================================================
    矩阵A:
    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8 10]]
    向量b: [1 2 3]
    
    7.1 基本线性代数运算
    矩阵行列式: -3.00
    矩阵迹: 16
    矩阵秩: 3
    
    7.2 特征值和特征向量
    特征值: [16.70749332 -0.90574018  0.19824686]
    特征向量:
    [[-0.22351336 -0.86584578  0.27829649]
     [-0.50394563  0.0856512  -0.8318468 ]
     [-0.83431444  0.4929249   0.48018951]]
    
    7.3 奇异值分解
    U矩阵形状: (3, 3)
    奇异值: [17.41250517  0.87516135  0.19686652]
    Vt矩阵形状: (3, 3)
    
    7.4 线性方程组解: [-3.33333333e-01  6.66666667e-01  3.17206578e-17]
    验证Ax: [1. 2. 3.]


## 8. 随机数生成


```python
print("\n" + "="*50)
print("8. 随机数生成")
print("="*50)

# 设置随机种子
np.random.seed(42)

print(f"\n8.1 基本随机数")
print(f"均匀分布[0,1): {np.random.random(5)}")
print(f"均匀分布[a,b): {np.random.uniform(1, 10, 5)}")
print(f"正态分布: {np.random.normal(0, 1, 5)}")
print(f"整数随机: {np.random.randint(1, 100, 5)}")

print(f"\n8.2 特殊分布")
print(f"指数分布: {np.random.exponential(2, 5)}")
print(f"泊松分布: {np.random.poisson(3, 5)}")
print(f"二项分布: {np.random.binomial(10, 0.3, 5)}")

print(f"\n8.3 随机选择和打乱")
data_to_shuffle = np.arange(10)
print(f"原数据: {data_to_shuffle}")
np.random.shuffle(data_to_shuffle)
print(f"打乱后: {data_to_shuffle}")

choices_rand = np.random.choice([1, 2, 3, 4, 5], size=10, replace=True, p=[0.1, 0.1, 0.2, 0.3, 0.3]) # Renamed
print(f"加权随机选择: {choices_rand}")
```

    
    ==================================================
    8. 随机数生成
    ==================================================
    
    8.1 基本随机数
    均匀分布[0,1): [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
    均匀分布[a,b): [2.40395068 1.52275251 8.79558531 6.41003511 7.3726532 ]
    正态分布: [-0.46947439  0.54256004 -0.46341769 -0.46572975  0.24196227]
    整数随机: [89 49 91 59 42]
    
    8.2 特殊分布
    指数分布: [0.09557922 7.28059911 0.5299408  0.18995463 1.92669135]
    泊松分布: [5 0 3 2 3]
    二项分布: [2 4 3 2 3]
    
    8.3 随机选择和打乱
    原数据: [0 1 2 3 4 5 6 7 8 9]
    打乱后: [4 2 7 0 6 3 5 8 9 1]
    加权随机选择: [5 5 4 4 5 4 3 3 2 1]


## 9. 数组条件操作


```python
print("\n" + "="*50)
print("9. 数组条件操作")
print("="*50)

data_cond = np.random.randn(5, 5) # Renamed
print(f"随机数据:\n{data_cond}")

# where函数
print(f"\n9.1 where函数")
result_cond = np.where(data_cond > 0, data_cond, 0)  # 正数保留，负数置零
print(f"正数保留，负数置零:\n{result_cond}")

# 多条件
condition_multi = np.where(data_cond > 1, 'high', np.where(data_cond > 0, 'medium', 'low'))
print(f"\n多条件分类:\n{condition_multi}")

# select函数
conditions_select = [data_cond < -1, (data_cond >= -1) & (data_cond < 1), data_cond >= 1]
choices_select = ['low', 'medium', 'high']
selected = np.select(conditions_select, choices_select)
print(f"\nselect函数结果:\n{selected}")
```

    
    ==================================================
    9. 数组条件操作
    ==================================================
    随机数据:
    [[-1.91328024 -1.87567677 -1.36678214  0.63630511 -0.90672067]
     [ 0.47604259  1.30366127  0.21158701  0.59704465 -0.89633518]
     [-0.11198782  1.46894129 -1.12389833  0.9500054   1.72651647]
     [ 0.45788508 -1.68428738  0.32684522 -0.08111895  0.46779475]
     [ 0.73612235 -0.77970188 -0.84389636 -0.15053386 -0.96555767]]
    
    9.1 where函数
    正数保留，负数置零:
    [[0.         0.         0.         0.63630511 0.        ]
     [0.47604259 1.30366127 0.21158701 0.59704465 0.        ]
     [0.         1.46894129 0.         0.9500054  1.72651647]
     [0.45788508 0.         0.32684522 0.         0.46779475]
     [0.73612235 0.         0.         0.         0.        ]]
    
    多条件分类:
    [['low' 'low' 'low' 'medium' 'low']
     ['medium' 'high' 'medium' 'medium' 'low']
     ['low' 'high' 'low' 'medium' 'high']
     ['medium' 'low' 'medium' 'low' 'medium']
     ['medium' 'low' 'low' 'low' 'low']]
    
    select函数结果:
    [['low' 'low' 'low' 'medium' 'medium']
     ['medium' 'high' 'medium' 'medium' 'medium']
     ['medium' 'high' 'low' 'medium' 'high']
     ['medium' 'low' 'medium' 'medium' 'medium']
     ['medium' 'medium' 'medium' 'medium' 'medium']]


## 10. 数组排序


```python
print("\n" + "="*50)
print("10. 数组排序")
print("="*50)

unsorted = np.random.randint(1, 100, (4, 5))
print(f"未排序数组:\n{unsorted}")

print(f"\n10.1 基本排序")
print(f"整体排序: {np.sort(unsorted, axis=None)}")
print(f"按行排序:\n{np.sort(unsorted, axis=1)}")
print(f"按列排序:\n{np.sort(unsorted, axis=0)}")

print(f"\n10.2 排序索引")
indices = np.argsort(unsorted, axis=1)
print(f"排序索引:\n{indices}")

print(f"\n10.3 部分排序")
partitioned = np.partition(unsorted, 2, axis=1)  # 第3小的元素
print(f"部分排序(第3小):\n{partitioned}")
```

    
    ==================================================
    10. 数组排序
    ==================================================
    未排序数组:
    [[51 63 96 52 96]
     [ 4 94 23 15 43]
     [29 36 13 32 71]
     [59 86 28 66 42]]
    
    10.1 基本排序
    整体排序: [ 4 13 15 23 28 29 32 36 42 43 51 52 59 63 66 71 86 94 96 96]
    按行排序:
    [[51 52 63 96 96]
     [ 4 15 23 43 94]
     [13 29 32 36 71]
     [28 42 59 66 86]]
    按列排序:
    [[ 4 36 13 15 42]
     [29 63 23 32 43]
     [51 86 28 52 71]
     [59 94 96 66 96]]
    
    10.2 排序索引
    排序索引:
    [[0 3 1 2 4]
     [0 3 2 4 1]
     [2 0 3 1 4]
     [2 4 0 3 1]]
    
    10.3 部分排序
    部分排序(第3小):
    [[51 52 63 96 96]
     [ 4 15 23 94 43]
     [13 29 32 36 71]
     [28 42 59 66 86]]


## 11. 集合操作


```python
print("\n" + "="*50)
print("11. 集合操作")
print("="*50)

set1 = np.array([1, 2, 3, 4, 5])
set2 = np.array([3, 4, 5, 6, 7])

print(f"集合1: {set1}")
print(f"集合2: {set2}")

print(f"\n11.1 集合运算")
print(f"并集: {np.union1d(set1, set2)}")
print(f"交集: {np.intersect1d(set1, set2)}")
print(f"差集: {np.setdiff1d(set1, set2)}")
print(f"对称差集: {np.setxor1d(set1, set2)}")

print(f"\n11.2 成员检测")
print(f"set1中的元素是否在set2中: {np.in1d(set1, set2)}")

# 去重
duplicates = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print(f"\n原数组: {duplicates}")
print(f"去重后: {np.unique(duplicates)}")
print(f"去重并返回计数: {np.unique(duplicates, return_counts=True)}")
```

    
    ==================================================
    11. 集合操作
    ==================================================
    集合1: [1 2 3 4 5]
    集合2: [3 4 5 6 7]
    
    11.1 集合运算
    并集: [1 2 3 4 5 6 7]
    交集: [3 4 5]
    差集: [1 2]
    对称差集: [1 2 6 7]
    
    11.2 成员检测
    set1中的元素是否在set2中: [False False  True  True  True]
    
    原数组: [1 2 2 3 3 3 4 4 4 4]
    去重后: [1 2 3 4]
    去重并返回计数: (array([1, 2, 3, 4]), array([1, 2, 3, 4]))


## 12. 广播机制


```python
print("\n" + "="*50)
print("12. 广播机制")
print("="*50)

# 不同形状数组的运算
print(f"\n12.1 广播示例")
a_bc = np.array([[1, 2, 3], [4, 5, 6]]) # Renamed
b_bc = np.array([10, 20, 30]) # Renamed
print(f"数组a (2x3):\n{a_bc}")
print(f"数组b (3,): {b_bc}")
print(f"广播相加:\n{a_bc + b_bc}")

# 更复杂的广播
c_bc = np.array([[1], [2]]) # Renamed
print(f"\n数组c (2x1):\n{c_bc}")
print(f"a + c:\n{a_bc + c_bc}")

# 广播规则演示
print(f"\n12.2 广播规则")
x_bc = np.arange(4) # Renamed
y_bc = np.arange(5).reshape(5, 1) # Renamed
print(f"x形状: {x_bc.shape}, y形状: {y_bc.shape}")
result_bc = x_bc + y_bc
print(f"广播结果形状: {result_bc.shape}")
print(f"广播结果:\n{result_bc}")
```

    
    ==================================================
    12. 广播机制
    ==================================================
    
    12.1 广播示例
    数组a (2x3):
    [[1 2 3]
     [4 5 6]]
    数组b (3,): [10 20 30]
    广播相加:
    [[11 22 33]
     [14 25 36]]
    
    数组c (2x1):
    [[1]
     [2]]
    a + c:
    [[2 3 4]
     [6 7 8]]
    
    12.2 广播规则
    x形状: (4,), y形状: (5, 1)
    广播结果形状: (5, 4)
    广播结果:
    [[0 1 2 3]
     [1 2 3 4]
     [2 3 4 5]
     [3 4 5 6]
     [4 5 6 7]]


## 13. 内存布局和性能


```python
print("\n" + "="*50)
print("13. 内存布局和性能")
print("="*50)

# C风格 vs Fortran风格
print(f"\n13.1 内存布局")
c_array = np.array([[1, 2, 3], [4, 5, 6]], order='C')
f_array = np.array([[1, 2, 3], [4, 5, 6]], order='F')

print(f"C风格数组: {c_array.flags['C_CONTIGUOUS']}")
print(f"Fortran风格数组: {f_array.flags['F_CONTIGUOUS']}")

# 视图 vs 副本
print(f"\n13.2 视图与副本")
original_mem = np.arange(10) # Renamed
view = original_mem[::2]  # 视图
copy = original_mem[::2].copy()  # 副本

print(f"原数组: {original_mem}")
print(f"视图: {view}")
print(f"副本: {copy}")

original_mem[0] = 999
print(f"修改原数组后:")
print(f"原数组: {original_mem}")
print(f"视图: {view}")
print(f"副本: {copy}")
```

    
    ==================================================
    13. 内存布局和性能
    ==================================================
    
    13.1 内存布局
    C风格数组: True
    Fortran风格数组: True
    
    13.2 视图与副本
    原数组: [0 1 2 3 4 5 6 7 8 9]
    视图: [0 2 4 6 8]
    副本: [0 2 4 6 8]
    修改原数组后:
    原数组: [999   1   2   3   4   5   6   7   8   9]
    视图: [999   2   4   6   8]
    副本: [0 2 4 6 8]


## 14. 结构化数组


```python
print("\n" + "="*50)
print("14. 结构化数组")
print("="*50)

# 定义结构化数据类型
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
people = np.array([('Alice', 25, 55.5), ('Bob', 30, 70.2), ('Charlie', 35, 80.1)], dtype=dt)

print(f"结构化数组:\n{people}")
print(f"姓名: {people['name']}")
print(f"年龄: {people['age']}")
print(f"体重: {people['weight']}")

# 按字段排序
sorted_by_age = np.sort(people, order='age')
print(f"\n按年龄排序:\n{sorted_by_age}")
```

    
    ==================================================
    14. 结构化数组
    ==================================================
    结构化数组:
    [('Alice', 25, 55.5) ('Bob', 30, 70.2) ('Charlie', 35, 80.1)]
    姓名: ['Alice' 'Bob' 'Charlie']
    年龄: [25 30 35]
    体重: [55.5 70.2 80.1]
    
    按年龄排序:
    [('Alice', 25, 55.5) ('Bob', 30, 70.2) ('Charlie', 35, 80.1)]


## 15. 高级技巧


```python
print("\n" + "="*50)
print("15. 高级技巧")
print("="*50)

# 向量化函数
print(f"\n15.1 向量化函数")
def python_func(x):
    if x > 0:
        return x ** 2
    else:
        return 0

vectorized_func = np.vectorize(python_func)
test_data_adv = np.array([-2, -1, 0, 1, 2]) # Renamed
print(f"输入: {test_data_adv}")
print(f"向量化结果: {vectorized_func(test_data_adv)}")

# 数组的字符串表示
print(f"\n15.2 数组显示控制")
large_array_adv = np.random.random((10, 10)) # Renamed
with np.printoptions(precision=2, suppress=True, threshold=50):
    print(f"格式化显示:\n{large_array_adv}")

# 数组保存和加载
print(f"\n15.3 数组保存和加载")
test_array_save = np.random.random((3, 3)) # Renamed
np.save('/tmp/test_array.npy', test_array_save)
loaded_array = np.load('/tmp/test_array.npy')
print(f"保存并加载的数组相等: {np.array_equal(test_array_save, loaded_array)}")

# 多个数组保存
np.savez('/tmp/multiple_arrays.npz', arr1=test_array_save, arr2=large_array_adv[:3, :3])
loaded_data = np.load('/tmp/multiple_arrays.npz')
print(f"加载的数组键: {list(loaded_data.keys())}")

print("\n" + "="*50)
print("Numpy高级示例完成！")
print("="*50)
```

    
    ==================================================
    15. 高级技巧
    ==================================================
    
    15.1 向量化函数
    输入: [-2 -1  0  1  2]
    向量化结果: [0 0 0 1 4]
    
    15.2 数组显示控制
    格式化显示:
    [[0.81 0.63 0.87 ... 0.81 0.9  0.32]
     [0.11 0.23 0.43 ... 0.42 0.22 0.12]
     [0.34 0.94 0.32 ... 0.96 0.25 0.5 ]
     ...
     [0.88 0.26 0.66 ... 0.09 0.9  0.9 ]
     [0.63 0.34 0.35 ... 0.64 0.08 0.16]
     [0.9  0.61 0.01 ... 0.55 0.69 0.65]]
    
    15.3 数组保存和加载
    保存并加载的数组相等: True
    加载的数组键: ['arr1', 'arr2']
    
    ==================================================
    Numpy高级示例完成！
    ==================================================


# Python从基础到进阶学习示例

本笔记包含了Python从基础到进阶的各种示例代码，适合学习和参考。

## 1. 基础数据结构和操作


```python
print("1. 基础数据结构和操作")
# 列表操作
l = [10, 20, 30, 40]
l.append(50)
l.extend([60,70])
print(f"添加元素: {l}")
l.insert(2, 25)
print(f"插入元素: {l}")
l.pop()
print(f"弹出元素: {l}")
print(f"切片操作: {l[1:4]}")
print(f"列表推导式: {[x*2 for x in l]}")
```

    1. 基础数据结构和操作
    添加元素: [10, 20, 30, 40, 50, 60, 70]
    插入元素: [10, 20, 25, 30, 40, 50, 60, 70]
    弹出元素: [10, 20, 25, 30, 40, 50, 60]
    切片操作: [20, 25, 30]
    列表推导式: [20, 40, 50, 60, 80, 100, 120]



```python
# 字典操作
d = {"name": "张三", "age": 25, "city": "北京"}
d["job"] = "开发工程师"  # 添加键值对
del d["age"]  # 删除键值对
print(f"字典视图: {d.keys()}, {d.values()}, {d.items()}")
```

    字典视图: dict_keys(['name', 'city', 'job']), dict_values(['张三', '北京', '开发工程师']), dict_items([('name', '张三'), ('city', '北京'), ('job', '开发工程师')])



```python
# 集合
s1 = {1, 2, 3}
s2 = {3, 4, 5}
print(f"集合并集: {s1 | s2}")
```

    集合并集: {1, 2, 3, 4, 5}


## 2. 函数进阶


```python
print("\n2. 函数进阶")
# lambda函数
square = lambda x: x*x
print(f"Lambda平方计算: {square(5)}")
```

    
    2. 函数进阶
    Lambda平方计算: 25



```python
# 闭包函数
def multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply
double = multiplier(2)
print(f"闭包函数应用: {double(8)}")
```

    闭包函数应用: 16


## 3. 面向对象编程


```python
print("\n3. 面向对象编程")
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def introduce(self):
        return f"我叫{self.name}，今年{self.age}岁"

# 创建实例
p = Person("李四", 30)
print(p.introduce())
```

    
    3. 面向对象编程
    我叫李四，今年30岁


## 4. 文件处理


```python
print("\n4. 文件处理")
# 写入文件
with open("demo.txt", "w") as f:
    f.write("Hello, File IO!")

# 读取文件
with open("demo.txt", "r") as f:
    content = f.read()
    print(f"文件内容: {content}")
```

    
    4. 文件处理
    文件内容: Hello, File IO!


## 5. 异常处理


```python
print("\n5. 异常处理")
try:
    result = 10 / 0
except ZeroDivisionError:
    print("捕获除零错误！")
```

    
    5. 异常处理
    捕获除零错误！


## 6. Python进阶特性


```python
print("\n6. Python进阶特性")
# 装饰器
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"正在执行函数: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    return a + b

print(f"装饰器应用结果: {add(3, 4)}")
```

    
    6. Python进阶特性
    正在执行函数: add
    装饰器应用结果: 7


## 7. 生成器示例


```python
print("\n7. 生成器示例")
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for i in countdown(5):
    print(f"倒计时: {i}")

print("所有示例运行完成！")
```

    
    7. 生成器示例
    倒计时: 5
    倒计时: 4
    倒计时: 3
    倒计时: 2
    倒计时: 1
    所有示例运行完成！


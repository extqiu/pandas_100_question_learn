#url:https://www.shiyanlou.com/courses/1091/learning/
print("		1. 导入 Pandas：")
import pandas as pd #导入 Pandas 模块

print('		2. 查看 Pandas 版本信息：')
print(pd.__version__) #查看 Pandas 版本信息

print('		3. 从列表创建 Series：')
arr = [0, 1, 2, 3, 4]
s1 = pd.Series(arr)  # 如果不指定索引，则默认从 0 开始
print(s1)

print('		4. 从 Ndarray 创建 Series：')
import numpy as np

n = np.random.randn(5)  # 创建一个随机 Ndarray 数组

index = ['a', 'b', 'c', 'd', 'e']
s2 = pd.Series(n, index=index)
print(s2)

print("		5. 从字典创建 Series：")
d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5} # 定义示例字典
s3 = pd.Series(d)
print(s3)

print("		6. 修改 Series 索引：")
print(s1)
s1.index = ['A', 'B', 'C', 'D', 'E'] # 修改S1的索引
print(s1)

print("		7. Series 纵向拼接：")
s4 = s3.append(s1) 	# 将 s1 拼接到 s3
print(s4)

print("		8. Series 按指定索引删除元素：")
print(s4)
s4 = s4.drop('e') 	# 删除索引为 e 的值
print(s4)

print("		9. Series 修改指定索引元素：")
print(s4)
s4['A'] = 6 		#修改索引为 A 的值 = 6
print(s4)

print(" 	10. Series 按指定索引查找元素：")
print (s4)
print(s4['B'])  #Series 按指定索引查找元素 	

print(" 	11. Series 切片操作：")
print (s4)
print (s4[:3]) 		#对s4的前 3 个数据访问

print(" 	12. Series 加法运算：")
print(s4)
print(s3)
print(s4.add(s3)) 	#Series 的加法运算是按照索引计算，如果索引不同则填充为 NaN（空值）

print(" 	13. Series 减法运算：")
print(s4)
print(s3)
print(s4.sub(s3)) 	#Series的减法运算是按照索引对应计算，如果不同则填充为 NaN（空值）。

print("         14. Series 乘法运算：")
print(s4)
print(s3)
print(s4.mul(s3)) 	#Series 的乘法运算是按照索引对应计算，如果索引不同则填充为 NaN（空值）。

print("         15. Series 除法运算：")
print(s4)
print(s3)
print(s4.div(s3)) 	#Series 的除法运算是按照索引对应计算，如果索引不同则填充为 NaN（空值）。

print(" 	16. Series 求中位数：")
print(s4)
print(s4.median())

print("         17. Series 求和：")
print(s4)
print(s4.sum())

print("         18. Series 求最大值：")
print(s4)
print(s4.max())

print("         19. Series 求最小值：")
print(s4)
print(s4.min())

print(" 	20. 通过 NumPy 数组创建 DataFrame：")
dates = pd.date_range('today', periods=6) # 定义时间序列作为 index
print(dates)
num_arr = np.random.randn(6, 4) # 传入 numpy 随机数组(6行4列)
print(num_arr)
columns = ['A', 'B', 'C', 'D'] # 将列表作为列名
df1 = pd.DataFrame(num_arr, index=dates, columns=columns)
print(df1)

print(" 	21. 通过字典数组创建 DataFrame：")
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

print(data)
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
print(labels)
df2 = pd.DataFrame(data, index=labels)
print(df2)

print(" 	22. 查看 DataFrame 的数据类型：")
print(df2.dtypes)

print("		23. 预览 DataFrame 的前 5 行数据：")
print(df2.head())

print(" 	24. 查看 DataFrame 的后 3 行数据：")
print(df2.tail(3))

print(" 	25.查看 DataFrame 的索引：")
print(df2.index)

print(" 	26. 查看 DataFrame 的列名：")
print(df2.columns)

print(" 	27. 查看 DataFrame 的数值：(不显示行列)")
print(df2)
print(df2.values)

print("         28. 查看 DataFrame 的统计数据：")
print(df2)
print(df2.describe())

print("         29. DataFrame 转置操作：(从2行3列数据变成3行2列数据)")
print(df2)
print(df2.T)

print("         30. 对 DataFrame 进行按列排序：(升序)")
print(df2)
print(df2.sort_values(by='age'))

print("         31. 对 DataFrame 数据切片：(只显示第二到第四行数据)")
print(df2)
print(df2[1:4]) 	#只显示第二到第四行数据

print("         32. 对 DataFrame 通过标签查询（单列）：")
print(df2)
print(df2['age'])
print(df2.age)

print("         33. 对 DataFrame 通过标签查询（多列）：")
print(df2)
print(df2[['age', 'animal']])

print("         34. 对 DataFrame 通过位置查询：(查询 2到3 行)")
print(df2)
print(df2.iloc[1:3]) 	#查询 2到3 行

print("         35. DataFrame 副本拷贝：")
print(df2)
df3 = df2.copy()
print(df3)

print("         36. 判断 DataFrame 元素是否为空：")
print(df3)
print(df3.isnull())

print("         37. 添加列数据：")
print(df3)
num = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=df3.index)
df3['No.'] = num   ## 添加以 'No.' 为列名的新数据列
print(df3)

print("         38. 根据 DataFrame 的下标值进行更改。：(# 修改第 2 行与第 2 列对应的值)")
print(df3)
df3.iat[1, 1] = 2 ## 修改第 2 行与第 2 列对应的值 3.0 → 2.0
print(df3)

print("         39. 根据 DataFrame 的标签对数据进行修改：(修改f行age列)")
print(df3)
df3.loc['f', 'age'] = 1.5
print(df3)

print("         40. DataFrame 求平均值操作：")
print(df3)
print(df3.mean())

print("         41. 对 DataFrame 中任意列做求和操作：")
print(df3)
print(df3['visits'].sum())


print("         42. 将字符串转化为小写字母：")
string = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca',
                    np.nan, 'CABA', 'dog', 'cat'])
print(string)
print(string.str.lower())

print("         43. 将字符串转化为大写字母：")
print(string)
print(string.str.upper())

print("         44. 对缺失值进行填充：")
df4 = df3.copy()
print(df4)
print(df4.fillna(value=3))


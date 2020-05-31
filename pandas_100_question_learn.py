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
print (s4[0:3]) 

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

print("         45. 删除存在缺失值的行：")
df5 = df3.copy()
print(df5)
print(df5.dropna(how='any'))  ## 任何存在 NaN 的行都将被删除


print("         46. DataFrame 按指定列对齐(key)：")
left = pd.DataFrame({'key': ['foo1', 'foo2'], 'one': [1, 2]})
right = pd.DataFrame({'key': ['foo2', 'foo3'], 'two': [4, 5]})

print(left)
print(right)

# 按照 key 列对齐连接，只存在 foo2 相同，所以最后变成一行
print(pd.merge(left, right, on='key'))

print(" 	47. CSV 文件写入：")
df3.to_csv('animal.csv')
print("写入成功.")

print("         48. CSV 文件读取：")
df_animal = pd.read_csv('animal.csv')
print(df_animal)

print(" 	49. Excel 写入操作：")
df3.to_excel('animal.xlsx', sheet_name='Sheet1')
print("写入成功.")

# -*- coding:utf-8 -*-
print(" 	50. Excel 读取操作：")
print(pd.read_excel('animal.xlsx', 'Sheet1', index_col=None, na_values=['NA']))

print(" 	51. 建立一个以 2018 年每一天为索引，值为随机数的 Series：")
dti = pd.date_range(start='2018-01-01', end='2018-12-31', freq='D')
# np.random.rand
# 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
s = pd.Series(np.random.rand(len(dti)), index=dti)
print(dti)
print(s)

print(" 	52. 统计s 中每一个周三对应值的和：")
print(s[s.index.weekday == 2])
print('\n')
print(s[s.index.weekday == 2].sum())

print(" 	53. 统计s中每个月值的平均值：")
print(s.resample('M'))
print('\n')
print(s.resample('M').mean())

print(" 	54. 将 Series 中的时间进行转换（秒转分钟）(不理解)：")
s = pd.date_range('today', periods=100, freq='S') #采集当前时间之后的100s freq频率为秒(second)
print(s)
print(len(s))
ts = pd.Series(np.random.randint(0, 500, len(s)), index=s)
# np.random.randint(0, 500, len(s)), index=s 生成一个100行矩阵.以时间为索引 索引内容为0到500
print(ts)
print(ts.resample('Min').sum())

print(" 	55. UTC 世界时间标准：(结果不正确 时间标准时间+8=北京时间)")
s = pd.date_range('today', periods=1, freq='D')  # 获取当前时间
print(s)
ts = pd.Series(np.random.randn(len(s)), s)  # 随机数值
print(ts)
ts_utc = ts.tz_localize('UTC')  # 转换为 UTC 时间
print("输出UTC 时间")
print(ts_utc)

print(" 	56. 转换为上海所在时区：(结果不正确)")
print(ts_utc.tz_convert('Asia/Shanghai'))

print(" 	57.不同时间表示方式的转换：")
rng = pd.date_range('1/1/2018', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
ps = ts.to_period()
print(ps)
print(ps.to_timestamp())

print(" 	58. 创建多重索引 Series：")
letters = ['A', 'B', 'C']
numbers = list(range(10))

mi = pd.MultiIndex.from_product([letters, numbers])  # 设置多重索引
print(mi)
s = pd.Series(np.random.rand(30), index=mi)  # 随机数
print(s)

print(" 	59. 多重索引 Series 查询：")
print(s.loc[:, [1, 3, 6]])   ## 查询索引为 1，3，6 的值


print(" 	60. 多重索引 Series 切片：")
print(s)
print(s.loc[pd.IndexSlice[:'B', 5:]])
print(s.loc[pd.IndexSlice['A':'B', 5:9]])

print(" 	61. 根据多重索引创建 DataFrame：")
frame = pd.DataFrame(np.arange(12).reshape(6, 2),
                     index=[list('AAABBB'), list('123123')],
                     columns=['hello', 'shiyanlou'])
print(frame)

print(" 	62. 多重索引设置列名称：")
frame.index.names = ['first', 'second']
print(frame)

print(" 	63. DataFrame 多重索引分组求和：")
print(frame.groupby('first').sum())

print(" 	64. DataFrame 行列名称转换：")
print(frame)
print(frame.stack())

print(" 	65. DataFrame 索引转换：")
print(frame)
print(frame.unstack())

print(" 	66. DataFrame 条件查找：(查找 age 大于 3 的全部信息)")
# 示例数据
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data, index=labels)
print(df)
print(df[df['age'] > 3])

print(" 	67. 根据行列索引切片：(查询第3.4行第2.3列数据)")
print(df)
print(df.iloc[2:4, 1:3]) 	#查询第3.4行第2.3列数据 前闭后开

print(" 	68. DataFrame 多重条件查询：(查找 age<3 且为 cat 的全部数据。)")
df = pd.DataFrame(data, index=labels)
print(df)
print(df[(df['animal'] == 'cat') & (df['age'] < 3)])

print(" 	69. DataFrame 按关键字查询：(查询猫狗数据)")
print(df3)
print(df3[df3['animal'].isin(['cat', 'dog'])])

print(" 	70. DataFrame 按标签及列名查询。：(查询第4.第5.第9行第age列和第animail列数据)")
print(df)
print(df.loc[df2.index[[3, 4, 8]], ['animal', 'age']]) #索引从0开始

print(" 	71. DataFrame 多条件排序：(按照 age 降序，visits 升序排列)")
print(df)
print(df.sort_values(by=['age', 'visits'], ascending=[False, True]))

print(" 	72.DataFrame 多值替换：(将 priority 列的 yes 值替换为 True，no 值替换为 False。)")
print(df)
print(df['priority'].map({'yes': True, 'no': False}))

print(" 	73. DataFrame 分组求和：")
print(df4)
print(df4.groupby('animal').sum())

print(" 	74. 使用列表拼接多个 DataFrame：")
temp_df1 = pd.DataFrame(np.random.randn(5, 4))  # 生成由随机数组成的 DataFrame 1
temp_df2 = pd.DataFrame(np.random.randn(5, 4))  # 生成由随机数组成的 DataFrame 2
temp_df3 = pd.DataFrame(np.random.randn(5, 4))  # 生成由随机数组成的 DataFrame 3

print(temp_df1)
print(temp_df2)
print(temp_df3)

pieces = [temp_df1, temp_df2, temp_df3]
print(pd.concat(pieces))

print(" 	75. 找出 DataFrame 表中和最小和最大的列：")
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
print(df)
print(df.sum().idxmin())  # idxmax(), idxmin() 为 Series 函数返回最大最小值的索引值
print(df.sum().idxmax())

print(" 	76. DataFrame 中每个元素减去每一行的平均值的值：")
df = pd.DataFrame(np.random.random(size=(2, 2)))
print("输出原始数据")
print(df)
print("输出每列的平均值")
print(df.mean())  #计算所有数的平均值
print("输出每列的平均值")
print(df.mean(axis=0))
print("输出每行的平均值")
print(df.mean(axis=1)) 
print("每个元素减去每一行的平均值的值") 
print(df.sub(df.mean(axis=1), axis=0)) #mean 平均数

print(" 	77. DataFrame 分组，并得到每一组中最大三个数之和：(a.b.c组)")
df = pd.DataFrame({'A': list('aaabbcaabcccbbc'),
                   'B': [12, 345, 3, 1, 45, 14, 4, 52, 54, 23, 235, 21, 57, 3, 87]})
print(df)
print("输出每组最大的三个数")
print(df.groupby('A')['B'].nlargest(3))
print("输出每组最大的三个数之和")
print(df.groupby('A')['B'].nlargest(3).sum(level=0))


print(" 	78. 透视表的创建：(新建表将 A, B 列作为索引进行聚合(聚合:默认用平均值聚合)。)")

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})
print(df)
print(pd.pivot_table(df, index=['A', 'B']))

print(" 	79. 透视表按指定行进行聚合：(将该 DataFrame 的 D 列聚合，按照 A,B 列为索引进行聚合，聚合的方式为默认求均值。)")
print(df)
print(pd.pivot_table(df, values=['D'], index=['A', 'B']))

print(" 	80. 透视表聚合方式定义：")
print(df)
print(pd.pivot_table(df, values=['D'], index=['A', 'B'], aggfunc=[np.sum, len]))  #len不理解

print(" 	81. 透视表利用额外列进行辅助分割：(D 列按照 A,B 列进行聚合时，若关心 C 列对 D 列的影响，可以加入 columns 值进行分析。)")
print(df)
print(pd.pivot_table(df, values=['D'], index=['A', 'B'],
               columns=['C'], aggfunc=np.sum))

print(" 	82. 透视表的缺省值处理：(在透视表中由于不同的聚合方式，相应缺少的组合将为缺省值，可以加入 fill_value 对缺省值处理。)")
print(df)
print(pd.pivot_table(df, values=['D'], index=['A', 'B'],
               columns=['C'], aggfunc=np.sum, fill_value=0))   #不理解

print(" 	83. 绝对型数据定义：")
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": [
                  'a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
print(df)

print(" 	84. 对绝对型数据重命名：")
print(df)
df["grade"].cat.categories = ["very good", "good", "very bad"]
print(df)

print(" 	85. 重新排列绝对型数据并补充相应的缺省值：")
print(df)
df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"])
print("重新排列后数据和之前一样")
print(df)

print(" 	86. 对绝对型数据进行排序：")
print(df)
print(df.sort_values(by="grade"))

print(" 	87. 对绝对型数据进行分组(统计)：")
print(df)
print(df.groupby("grade").size())

print(" 	88. 缺失值拟合：(缺失值为前后值的平均值)")
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
                               'Budapest_PaRis', 'Brussels_londOn'],
                   'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
                   'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
                               '12. Air France', '"Swiss Air"']})
print(df)
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int) 
print(df)

print(" 	89. 数据列拆分：(From_to应该为两独立的两列From和To.且创建一个新表)")
temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']
print(df)
print(temp)

print(" 	90. 字符标准化：(capitalize.首字母大写)")
print(temp)
temp['From'] = temp['From'].str.capitalize()
temp['To'] = temp['To'].str.capitalize()
print(temp)

print(" 	91. 删除坏数据加入整理好的数据：")
print(df)
df = df.drop('From_To', axis=1) #From_to 列删除
df = df.join(temp) 		#加入整理好的 From 和 to 列
print(df)

print(" 	92. 去除多余字符：")
print(df)
df['Airline'] = df['Airline'].str.extract(
	'([a-zA-Z\s]+)', expand=False).str.strip() 	#正则表达式不理解
print(df)

print(" 	93. 格式规范：")
#在 RecentDelays 中记录的方式为列表类型，由于其长度不一，这会为后期数据分析造成很大麻烦。这里将 RecentDelays 的列表拆开，取出列表中的相同位置元素作为一列，若为空值即用 NaN 代替。
print(df)
delays = df['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_{}'.format(n)
                  for n in range(1, len(delays.columns)+1)]
df = df.drop('RecentDelays', axis=1).join(delays)
print(df)

print(" 	94. 信息区间划分：")
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Candy', 'Dany', 'Ella',
                            'Frank', 'Grace', 'Jenny'],
                   'grades': [58, 83, 79, 65, 93, 45, 61, 88]})

print(df)
def choice(x):
    if x > 60:
        return 1
    else:
        return 0
df.grades = pd.Series(map(lambda x: choice(x), df.grades))
print(df)

print(" 	95. 数据去重：")
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
print(df)
print(df.loc[df['A'].shift() != df['A']])  #尝试将 A 列中连续重复的数据清除

print(" 	96. 数据归一化：")
# 其中，Max-Min 归一化是简单而常见的一种方式，公式如下:
# Y = (X-Xmin)/(Xmax-Xmin)
def normalization(df):
    numerator = df.sub(df.min()) #sub减法
    denominator = (df.max()).sub(df.min())
    Y = numerator.div(denominator)  #除法
    return Y


df = pd.DataFrame(np.random.random(size=(5, 3)))
print("输出原值")
print(df)
print("输出第n列最小值")
print(df.min())
print("输出第n列最大值")
print(df.max())
print("正常值-所在列小值")
print(df.sub(df.min()))
print("所在列最大值-所在列最小值")
print((df.max()).sub(df.min()))
print("输出数据归一化值")   #类似矩阵除法
print(normalization(df))  

print(" 	97. Series 可视化:")
ts = pd.Series(np.random.randn(100), index=pd.date_range('today', periods=100))
ts = ts.cumsum()
print(ts.plot())

print(" 	98. DataFrame 折线图：")
df = pd.DataFrame(np.random.randn(100, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
print(df.plot())

print(" 	99. DataFrame 散点图：")
df = pd.DataFrame({"xs": [1, 5, 2, 8, 1], "ys": [4, 2, 1, 9, 6]})
df = df.cumsum()
print(df.plot.scatter("xs", "ys", color='red', marker="*"))

print(" 	100. DataFrame 柱形图：")
df = pd.DataFrame({"revenue": [57, 68, 63, 71, 72, 90, 80, 62, 59, 51, 47, 52],
                   "advertising": [2.1, 1.9, 2.7, 3.0, 3.6, 3.2, 2.7, 2.4, 1.8, 1.6, 1.3, 1.9],
                   "month": range(12)
                   })

ax = df.plot.bar("month", "revenue", color="yellow")
print(df.plot("month", "advertising", secondary_y=True, ax=ax))

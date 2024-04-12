import numpy as np
import pandas as pd
import csv
# 读取文本文件，以空格分隔每一行的内容，并创建DataFrame

data_txtDF = pd.read_csv("/InterKG/dataset/last-fm/kg_final.txt", sep=' ', header=None, names=['h', 'r', 't'])

# 按条件筛选待删除的行索引
row_indexes = data_txtDF[(data_txtDF['h'] > 48122) & (data_txtDF['t'] > 48122)].index

df_drop = data_txtDF.drop(row_indexes)

df1 = df_drop.pivot_table(values='t',columns='r',index='h',aggfunc=np.min)

df1.to_csv('final.csv',sep=',',index=False,header=False)








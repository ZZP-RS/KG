import numpy as np
import pandas as pd
import json
import itertools
user_dict = dict()  # 创建一个空字典存储用户交互记录 # key is user_id, value is interactions list
with open("/InterKG/dataset/last-fm/train.txt", "r") as file:
    for line in file:
        line = line.strip().split()
        user_id = line[0]
        items = list(set(line[1:]))
        user_dict[user_id] = items


cluster_dict = {}  # 创建一个空字典存储物品所在类别  # key is cluster_id, value is items list
with open('/InterKG/REG-cluster/cluster-200.txt', 'r') as file:
    lines = file.readlines()
for line in lines:
    line = line.strip()
    if line.startswith('Cluster'):
        category, items = line.split(':')
        cluster_dict[category] = items.split(',')


item_dict = dict()   # 创建一个空字典存储物品-用户交互记录  # key is item_id, value is user_id list
for user_id, interactions in user_dict.items():
    for item_id in interactions:
        if item_id not in item_dict:
            item_dict[item_id] = [user_id]
        else:
            item_dict[item_id].append(user_id)


#对每个类中的物品进行D分割
D_seg_in = dict()
D_seg_in = {user_id: [] for user_id in user_dict.keys()}
D_seg_out = dict()
D_seg_out = {user_id: [] for user_id in user_dict.keys()}
D_seg = dict()    #key is user_id value is D-seg
D_seg = {user_id: [] for user_id in user_dict.keys()}
for cluster_id in cluster_dict.keys():
    i=0
    D_seg_in = {user: [] for user in user_dict.keys()}
    D_seg_out = {user: [] for user in user_dict.keys()}
    for item in cluster_dict[cluster_id]:
        for user in user_dict.keys():
            if user in item_dict[item]:
                D_seg_in[user].append(item)
            else:
                D_seg_out[user].append(item)
            D_seg[user] = [list(D_seg_in[user]), list(D_seg_out[user])]
        i=i+1  #当前类中的item
        print(cluster_id,i,item)
    print("D_seg_in:", D_seg_in)
    print("D_seg_out:", D_seg_out)
    print("D_seg:", D_seg)
 # 将每次循环输出的D_seg写入文件（每一行代表一个类，第一行是cluster 0）
    with open("D_seg.json", "a") as f:
        json.dump(D_seg, f)
        f.write(",\n")  #

#建立每个聚类df，df大小都一样大,不是该类的item的行都赋值为0。
df = pd.read_csv('final.csv', sep=',', header=None)
df.replace(0, 1, inplace=True)  #为了区分将原始的空值赋值为1
df_copy = df.copy()
df_dict = {}  # 创建一个空字典,用来存储不同聚类的df   key is cluster_id, value is df_copy
for cluster_id in cluster_dict.keys():
    df_copy = df.copy()

    item = [int(i) for i in cluster_dict[cluster_id]]

    df_copy.iloc[~df_copy.index.isin(item), :] = 0

    df_dict[cluster_id] = df_copy

for cluster_id, df_copy in df_dict.items():
    print(f"cluster_id: {cluster_id}, df_copy: {df_copy}")


#对每两个关系下的物品进行P1分割，以及保留R的情况
group_dict={}  # key is group_name(元组)，value is item（group_df.index）
group_keys = set() #group_name(元组)的集合，用来选取R
p1_seg_dict ={}  #key is 关系编号i value is P1_list
for df_copy in df_dict.values():
    with open("p1_seg_dict.json", "a") as f:
        f.write("{")
    i=-1  #i是为所选的两个关系编个号
    # 根据任意两列的数据对 item 进行分组
    for col1, col2 in itertools.combinations(df_copy.columns, 2):#选择任意两列数据 Cn2次
        i=i+1
        group_dict = {}
        group_keys = set()
        p1_seg_dict ={}
        print('两种关系：',col1, col2) #输出所选择的两列（即哪两个关系）
        print('两种关系对应的编号', i)
        groups = df_copy.groupby([col1, col2])  #根据这两列的值进行分组
        # 遍历每个分组，输出每个分组中的 item
        for group_name, group_df in groups:
            group_key = group_name # 记录分组结果的键,group_name是元组的形式
            group_value = list(group_df.index)  # 记录分组结果的值，列表
            if group_key == (0, 0):  # 删除键为(0, 0)的，即不属于该类的
                group_dict.pop(group_key, None)
            else:
                group_dict[group_key] = group_value
            group_keys.add(group_key)  # 将 group_key 添加到集合中
            group_keys.discard((0,0))
        print('group_dict',group_dict)
        print('group_keys:', group_keys)
        P1_list = [list(v) for v in group_dict.values()]  #P1_list是将p1的分组结果写成list类型，其中每个元素都是list
        with open("p1_seg_dict.json", "a") as f:
            f.write(f'"{i}":{json.dumps(P1_list)}')
            f.write(",")
    with open("p1_seg_dict.json", "a") as f:
        f.seek(f.tell() - 1, 0)
        f.truncate()
        f.write("},\n")
        print('P1_list:',P1_list)


#计算D分割和P1分割的相似度 用到两个文件：p1_seg_dict.json和D_seg.json
with open('D_seg.json', 'r') as f_d, open('p1_seg_dict_update.json', 'r') as f_p1:
    for line_d, line_p1 in zip(f_d, f_p1):
        d_seg_dict = json.loads(line_d.strip(",\n"))
        p1_seg_dict = json.loads(line_p1.strip(",\n"))
        for key1 in d_seg_dict:
            d_set = set([frozenset(region) for region in d_seg_dict[key1]])
            for key2 in p1_seg_dict:
                p_set = set([frozenset(region) for region in p1_seg_dict[key2]])
                intersection = set()
                for dj in d_set:
                    for pi in p_set:
                        if pi & dj:
                            intersection.add(pi & dj)
                numerator = len(p_set) + len(d_set)
                denominator = 2 * len(intersection)
                similarity = numerator / denominator if denominator else 0
                print(similarity)
                #将相似度结果写入文件中，每一行是一个类的结果，相似度顺序依次为每个用户下的D_seg*每两种关系下的p1_seg  每行数量为：用户数*关系对数
                with open("D-P1_similarity.txt", "a") as file:
                    file.write(str(similarity) + " ")
        with open("D-P1_similarity.txt", "a") as file:
            file.write("\n")







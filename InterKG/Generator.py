import json
i=0

# 将p1_seg_dict.json和D_seg.json中的列表弄成统一的格式
with open('p1_seg_dict.json', 'r') as f_d:
    with open('p1_seg_dict_update.json', 'a') as f_update:
        for line_d in f_d:
            d_seg_dict = json.loads(line_d.strip(",\n"))
            for key, value in d_seg_dict.items():

                for i in range(len(value)):

                    value[i] = [str(item) if isinstance(item, int) else [str(subitem) for subitem in item] for item in
                                value[i]]

            json.dump(d_seg_dict, f_update)
            f_update.write(",\n")


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

with open('D_seg.json', 'r') as f_d, open('p1_seg_dict_update.json', 'r') as f_p1:
    for line_d, line_p1 in zip(f_d, f_p1):
        i=i+1
        d_seg_dict = json.loads(line_d.strip(",\n"))
        p1_seg_dict = json.loads(line_p1.strip(",\n"))
        for key2, value2 in p1_seg_dict.items():
            p_list = value2
            print(key2)  # 输出当前的两种关系对应编号
            result_dict = {} # 新的P2分割的字典，键是用户，键值是新分割的列表

            max_similarity = 0  # 初始化最大的similarity值为0
            max_key1 = None  # 初始化最大similarity值对应的key1为None
            min_list_length = float('inf')  # 初始化最小列表长度为无穷大
            max_dict = {}

            for key1, value1 in d_seg_dict.items():
                d_set = value1[0] #D分割的第一个小列表，也就是交互的项目
                com_lists = []  #新的P2分割的列表

                for sublist in p_list: #p1分割中遍历每一个小列表
                    if any(element in sublist for element in d_set):
                        com_lists.extend(sublist)
                if com_lists:
                    if key1 not in result_dict:
                        result_dict[key1] = []
                    result_dict[key1].extend(com_lists)
                    # print(key1,com_lists)
                    d_set2 = set(d_set)
                    com_set = set(com_lists)
                    similarity = jaccard_similarity(d_set2, com_set)
                    # print(key1, similarity)
                    if similarity > max_similarity :
                        max_similarity = similarity
                        max_key1 = key1
                        min_list_length = len(com_lists)
                        max_dict = {max_key1: com_lists}

            # print("Key with maximum similarity :", max_key1)
            # print("Maximum similarity value:", max_similarity)
            # print(max_dict)
            with open(f"max_dict.txt", "a") as f:
                f.write(str(list((max_dict.values()))))
        with open(f"max_dict.txt", "a") as f:
            f.write('\n')



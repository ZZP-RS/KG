import re
filename = "max_dict.txt"  # 文件名
j=1
with open("new", "w") as output_file:
    with open(filename, "r") as file:
        for line in file:
            results = []
            lists = re.findall(r"\[\[(.*?)\]\]", line)  # 查找每个[[]]中的内容
            line_lists = []

            for i, lst in enumerate(lists):
                numbers = re.findall(r"\d+", lst)  # 提取数字元素
                line_lists.append(numbers)
                output_file.write(f"{numbers} % {i + 1 + 8} % {i + j + 106388}\n")

            j = j + 36

# 读取输入文件
with open('new', 'r') as file:
    lines = file.readlines()
    # 处理每一行的内容并写入输出文件
    with open('new-kg.txt', 'w') as file:
        for line in lines:
            # 获取每行的数字列表values[0]、关系values[1]、实体values[2]
            values = line.split(' %')
            nums = re.findall(r"\d+", values[0])
            for num in nums:
                output_line = f"{num}{values[1]}{values[2]}"
                file.write(output_line)

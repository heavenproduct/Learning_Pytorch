import os

# 拿到「当前脚本所在目录往上两级」的绝对路径
current_dir = os.getcwd()    # 1. os.getcwd() 获取当前工作目录
print(current_dir)

parent_dir = os.path.join(current_dir, "../..")    # 2. os.path.join() 拼接路径，'..' 表示上一级目录
print(parent_dir)                                  # 这里只是简单的拼接

data_root = os.path.abspath(parent_dir)    # 3. os.path.abspath() 转换为绝对路径
print(data_root)                           # 在这里获取地址的时候,会按照'..'表示上一级目录来处理传入的地址,实际效果就是返回上两级目录的地址

#!/bin/bash

# 原始文件名称
original_file="submit.sh"

# 复制文件的数量
copies=135

for i in $(seq 1 $copies); do
    # 创建新文件的名称
    new_file="submit_$i.sh"

    # 复制原始文件到新文件
    # cp $original_file $new_file

    # 在第10行替换文本
    sed -i "28s/2/4/" $new_file
    sed -i "29s/5/10/g" $new_file
done

#!/bin/bash

# 原始文件名称
original_file="submit_sbatch.sh"

# 复制文件的数量
copies=50

for i in $(seq 1 $copies); do
    # 创建新文件的名称
    new_file="submit_sbatch_$i.sh"


    # 在第10行替换文本
    sed -i "29s/100/1/" $new_file
done


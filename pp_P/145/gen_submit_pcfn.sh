#!/bin/bash

# 原始文件名称
original_file="./submit_pcfn.sh"

# 复制文件的数量
copies=149

for i in $(seq 0 $copies); do
    # 创建新文件的名称
    new_file="submit_pcfn_$i.sh"

    # 复制原始文件到新文件
    cp $original_file $new_file

    # 在第10行替换文本
    sed -i "43s/run_pcfn.py/run_pcfn_$i.py/" $new_file
done






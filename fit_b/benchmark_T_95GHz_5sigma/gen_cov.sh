#!/bin/bash

# 原始文件名称
original_file="./pix_cov_qu.py"

# 复制文件的数量
copies=20

for i in $(seq 0 $copies); do
    # 创建新文件的名称
    new_file="run_$i.py"

    # 复制原始文件到新文件
    cp $original_file $new_file

    # 在第10行替换文本
    sed -i "9s/flux_idx=0/flux_idx=$i/" $new_file
done





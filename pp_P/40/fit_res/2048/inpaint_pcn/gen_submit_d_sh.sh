#!/bin/bash

# 原始文件名称
original_file="./submit_d.sh"

# 复制文件的数量
copies=99

for i in $(seq 0 $copies); do
    # 创建新文件的名称
    new_file="submit_d_$i.sh"

    # 复制原始文件到新文件
    cp $original_file $new_file

    # 在第10行替换文本
    sed -i "43s/cvt_QU_to_EB_0/cvt_QU_to_EB_$i/" $new_file
done

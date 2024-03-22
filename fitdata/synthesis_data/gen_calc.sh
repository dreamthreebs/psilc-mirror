#!/bin/bash

# 基础文件路径，需要提前准备好
base_file="coadd.py"

# 循环从0到49，生成50个文件
for i in {0..49}; do
  # 计算起始和结束的值
  let start=$i*2
  let end=$start+2

  # 生成新的文件名
  new_file="calc_${i}.py"

  # 复制基础文件为新文件
  cp $base_file $new_file

  # 修改新文件的指定行
  # 注意：-i'' 对于GNU sed是直接修改文件，但在macOS下需要提供一个扩展名或者传递空字符串
  sed -i'' "10s/.*/n_rlz_begin = $start/" $new_file
  sed -i'' "11s/.*/n_rlz_end = $end/" $new_file
done


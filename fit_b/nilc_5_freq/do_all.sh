#!/bin/bash
# ============================
# batch_submit.sh
# ============================
MAX_JOBS=100       # 一次最多提交的作业数
TOTAL_JOBS=201     # 总作业数
CHECK_INTERVAL=5   # 每隔几秒检查一次队列状态

# ============================
# 提交循环
# ============================
for ((i=1; i<TOTAL_JOBS; i++)); do
    # 检查当前队列中的作业数（包括R和PD）
    current_jobs=$(squeue -u $USER -h | wc -l)

    # 如果当前作业数达到上限，则等待直到作业全部结束
    if (( current_jobs >= MAX_JOBS )); then
        echo "$(date): 当前作业数=$current_jobs 已达上限($MAX_JOBS)，等待所有作业完成..."
        while true; do
            running_jobs=$(squeue -u $USER -h | wc -l)
            if (( running_jobs == 0 )); then
                echo "$(date): 所有作业已完成，继续提交新一批..."
                break
            fi
            sleep $CHECK_INTERVAL
        done
    fi

    # 提交作业
    sbatch submit_run_${i}.sh
    echo "$(date): 提交作业 submit_run_${i}.sh"

    # 防止sbatch触发调度频率限制（轻微延迟）
    # sleep 0.1
done

echo "✅ 所有 $TOTAL_JOBS 个作业已提交完毕。"


#! /bin/bash

#=====================================================
#===== Modify the following options for your job =====
#=====    DON'T remove the #! /bin/bash lines    =====
#=====      DON'T comment #SBATCH lines          =====
#=====        of partition,account and           =====
#=====                qos                        =====
#=====================================================

# Specify the partition name from which resources will be allocated  
#SBATCH --partition=ali

# Specify which expriment group you belong to.
# This is for the accounting, so if you belong to many experiments,
# write the experiment which will pay for your resource consumption
#SBATCH --account=alicpt

# Specify which qos(job queue) the job is submitted to.
#SBATCH --qos=regular


# ====================================
#SBATCH --job-name=wym

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
#SBATCH --exclude=aliws005
# SBATCH --mem-per-cpu=2000
# SBATCH --nodelist=aliws010

#SBATCH -o /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/o%j.log
#SBATCH --error=/afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/e%j.log

#or use relative path(some example are listed)
# mpiexec python -u tod_gen4cc.py
# mpirun -np 7 ./cosmomc test.ini
# python as.py

date +%m-%d_%H-%M
threshold=3
number="74"
mkdir -p output
mkdir -p output_n
# mrs_alm_inpainting -v ./input/pcn/2sigma/${number}.fits ./mask/pcn/2sigma/${number}.fits ./output/pcn/2sigma/${number}.fits
mrs_alm_inpainting -v ./input/${number}.fits ./mask/mask.fits ./output/${number}.fits
mrs_alm_inpainting -v ./input_n/${number}.fits ./mask/mask.fits ./output_n/${number}.fits


date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)

# mv /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/output*.log /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/out@${DATE}.txt
# mv /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/error*.log /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/err@${DATE}.txt









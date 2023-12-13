#! /bin/bash
#SBATCH --partition=ali
#SBATCH --account=alicpt
#SBATCH --qos=regular
#SBATCH --job-name=wym
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6GB
#SBATCH --exclude=aliws[021-048],aliws005

#SBATCH -o /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/o%j.log
#SBATCH --error=/afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/e%j.log

date +%m-%d_%H-%M
mpiexec python -u /afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/test/proj_to_gnom/Cov_generator/Cov_size81reso1.5555555555555556.py
date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)

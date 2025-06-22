#!/bin/bash

# 添加SLURM路径
export PATH=$PATH:/work/apps/slurm/bin

#SBATCH --job-name=jittor-test
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 加载Conda环境配置，必须
source /work/apps/miniconda3/etc/profile.d/conda.sh
# 激活环境
conda activate jittor_comp_human
python3 -m jittor.test.test_example
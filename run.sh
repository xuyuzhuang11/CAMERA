#!/bin/bash
#SBATCH -J me
#SBATCH -o /home/yzxu/MoE/d16b/0.21_energycut/test.out
#SBATCH -e /home/yzxu/MoE/d16b/0.21_energycut/test.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -w gpu04
#SBATCH -c 8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=200G

. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.4.1

# export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

python d16b.py /home/share/models/deepseek-moe-16b-base wikitext2 --seed 1234 --nsamples 128 --calib-length 2048 \
  --save /home/yzxu/MoE/d16b/0.21_energycut \
  --func prune --pratio 0.21 --chunk-size 32 --reduce_ratio 0.90

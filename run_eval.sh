#!/bin/bash
#SBATCH -J bench
#SBATCH -o /home/yzxu/MoE/test_tasks.out
#SBATCH -e /home/yzxu/MoE/test_tasks.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 36:00:00
#SBATCH -w gpu09
#SBATCH -c 8
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --mem=128G

. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.4.1

# export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# python eval_ppl.py
python eval_tasks.py

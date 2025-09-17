#! /bin/bash

#SBATCH --account=arm
#SBATCH --partition=arm-interactive
#SBATCH --nodelist=arm2
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8192

echo "NUMBER OF GPUS = 1"

CONFIG=$1
EXP_NAME=$2

srun python3 train_ViT_lightning.py --config $CONFIG --exp_name $EXP_NAME
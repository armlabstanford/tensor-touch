GPUS=$1

# cpus = 16 * GPUS
CPUS=$((16 * $GPUS))

echo "NUMBER OF GPUS = ${GPUS}"
echo "NUMBER OF CPUS = ${CPUS}"

srun --account=arm --partition=arm-interactive --nodelist=arm2 -c ${CPUS} --mem-per-cpu=8192 --gres=gpu:${GPUS} --x11 --pty --time=50:00:00 \
        python3 train_ViT_lightning.py --epochs 100 --gpus ${GPUS} --batch_size ${CPUS} --num_workers ${CPUS} \
                        --exp_name exp/ViT_L1Loss_Full
python3 train_ViT_lightning.py --config configs/dt_ultra.yaml \
--ckpt_path 'mae_random_crop_hiera/checkpoints/last.ckpt' \
--finetune --exp_name exp/hiera_final_robust_mae \
--dataset_dir /arm/u/maestro/Desktop/DenseTact-Model/es4t/es4t/dataset_local/
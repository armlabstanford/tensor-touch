
# # python3 train_ViT_lightning.py --config configs/densenet_real_all.yaml \
# # --ckpt_path 'exp/densenet_real_depth_finetune_10_22/checkpoints/dt_model-epoch=94-AUC=71.73.ckpt' \
# # --finetune --exp_name exp/densenet_real_disp_force_stress_11_18_l1_gradient_cosine_1_lr_sched_1e-4  \
# # --encoder_paths 'exp/densenet_real_stress_pretrain_11_11/checkpoints/dt_model-epoch=94-AUC=68.07.ckpt' 'exp/densenet_real_force_11_11/checkpoints/dt_model-epoch=76-AUC=62.60.ckpt' 'exp/densenet_real_disp_AdamW_pretrain_11_11/checkpoints/dt_model-epoch=71-AUC=2.28.ckpt' \



# python3 train_ViT_lightning.py --config configs/densenet_real_all.yaml \
# --ckpt_path 'exp/densenet_real_depth_finetune_10_22/checkpoints/dt_model-epoch=94-AUC=71.73.ckpt' \
# --match_features_from_encoders  --finetune --exp_name exp/densenet_real_disp_stress_11_26_l1_upsample_theia \
# --encoder_paths 'exp/densenet_real_stress_pretrain_11_11/checkpoints/dt_model-epoch=94-AUC=68.07.ckpt' 'exp/densenet_real_force_11_11/checkpoints/dt_model-epoch=76-AUC=62.60.ckpt' 'exp/densenet_real_disp_AdamW_pretrain_11_11/checkpoints/dt_model-epoch=71-AUC=2.28.ckpt' \


python3 train_ViT_lightning.py --config configs/densenet_real_all.yaml \
--ckpt_path 'exp/densenet_real_depth_finetune_10_22/checkpoints/dt_model-epoch=94-AUC=71.73.ckpt' \
--match_features_from_encoders  --finetune --exp_name exp/densenet_real_disp_stress_12_4_l1_upsample_theia \
--encoder_paths 'exp/densenet_real_stress_pretrain_11_11/checkpoints/dt_model-epoch=94-AUC=68.07.ckpt' 'exp/densenet_real_disp_AdamW_pretrain_11_11/checkpoints/dt_model-epoch=71-AUC=2.28.ckpt' \

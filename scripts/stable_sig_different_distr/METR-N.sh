TRAIN_DIR=generated_images/imgs_w
VAL_DIR=val_mock

CHECKPOINT_NAME=metr_n
EVAL_PATH="$CHECKPOINT_NAME/imgs_no_w"

# Fine-tuning:

accelerate launch -m metr.finetune_ldm_decoder --num_keys 1 \
  --ldm_config v2-inference.yaml \
  --ldm_ckpt v2-1_512-ema-pruned.ckpt \
  --msg_decoder_path dec_48b_whit.torchscript.pt \
  --train_dir $TRAIN_DIR \
  --val_dir $VAL_DIR \
  --with_tracking \
  --project_name finetune_ldm_decoder \
  --run_name test \
  --output_dir finetune_ldm_decoder \
  --batch_size 4 \
  --steps 50 \
  --num_val_imgs 1 \
  --not_rand_key \
  --key_str 111010110101000001010111010011010100010000100111 \
  --checkpoint_name $CHECKPOINT_NAME

# Generation:

accelerate launch -m metr.run_metr \
  --project_name generate_$CHECKPOINT_NAME \
  --run_name 3k_4k_generation --w_channel 3 --w_pattern ring \
  --start 3000 --end 4000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --save_locally \
  --local_path $CHECKPOINT_NAME \
  --decoder_state_dict_path finetune_ldm_decoder/$CHECKPOINT_NAME.pth

# Evaluation:

accelerate launch -m metr.metr_pp_eval_stable_sig \
  --with_tracking \
  --project_name eval_$CHECKPOINT_NAME  \
  --run_name vae_st_tr_imgs_no_tree \
  --eval_imgs False --eval_bits True \
  --img_dir $EVAL_PATH \
  --output_dir logs_vae_st_tr_imgs_st_tr \
  --msg_decoder_path dec_48b_whit.torchscript.pt \
  --attack_mode few \
  --key_str 111010110101000001010111010011010100010000100111

# accelerate launch -m metr.run_metr \
#   --project_name for_paper_tests \
#   --run_name no_attack --w_channel 3 --w_pattern ring \
#   --start 0 --end 200 \
#   --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
#   --with_tracking \
#   --msg_type binary \
#   --use_random_msgs \
#   --save_locally \
#   --w_radius 10 \
#   --msg_scaler 100 \
#   --no_stable_sig \
#   --local_path generated_images

# accelerate launch -m metr.run_metr \
#   --project_name for_paper_tests \
#   --run_name diff_150 --w_channel 3 --w_pattern ring \
#   --start 0 --end 1 \
#   --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
#   --with_tracking \
#   --w_radius 10 \
#   --msg_type binary \
#   --use_random_msgs \
#   --msg_scaler 100 \
#   --no_stable_sig \
#   --use_attack \
#   --attack_type diff \
#   --diff_attack_steps 150

# TRAIN_DIR=fid_outputs/coco/ground_truth
# VAL_DIR=generated_images/imgs_w

#   accelerate launch -m metr.finetune_ldm_decoder --num_keys 1 \
#   --ldm_config v2-inference.yaml \
#   --ldm_ckpt v2-1_512-ema-pruned.ckpt \
#   --msg_decoder_path dec_48b_whit.torchscript.pt \
#   --train_dir $TRAIN_DIR \
#   --val_dir $VAL_DIR \
#   --with_tracking \
#   --project_name finetune_ldm_decoder \
#   --run_name test \
#   --output_dir finetune_ldm_decoder \
#   --batch_size 4 \
#   --steps 100 \
#   --num_val_imgs 200 \
#   --num_keys 1 \
#   --not_rand_key \
#   --key_str 111010110101000001010111010011010100010000100111

# -------------------------

# VAE_DECODER_PATH=finetune_ldm_decoder/ldm_decoder_checkpoint_000.pth

# accelerate launch -m metr.run_metr \
#   --project_name for_paper_tests \
#   --model_id stabilityai/stable-diffusion-2-1-base \
#   --run_name no_attack --w_channel 3 --w_pattern ring \
#   --start 0 --end 100 \
#   --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
#   --with_tracking \
#   --msg_type binary \
#   --use_random_msgs \
#   --w_radius 10 \
#   --msg_scaler 100 \
#   --save_locally \
#   --local_path metr_pp_generated_images \
#   --decoder_state_dict_path $VAE_DECODER_PATH

# -------------------

# EVAL_FOLDER=metr_pp_generated_images/imgs_w

# accelerate launch -m metr.metr_pp_eval_stable_sig \
#   --with_tracking \
#   --project_name for_paper_tests \
#   --run_name st_sig_eval \
#   --eval_imgs False --eval_bits True \
#   --img_dir $EVAL_FOLDER \
#   --output_dir eval_st_sig_logs \
#   --msg_decoder_path dec_48b_whit.torchscript.pt \
#   --attack_mode none \
#   --key_str 111010110101000001010111010011010100010000100111

#----------------

  # accelerate launch -m metr.run_metr_fid \
  # --project_name for_paper_tests \
  # --model_id stabilityai/stable-diffusion-2-1-base \
  # --run_name no_attack --w_channel 3 --w_pattern ring \
  # --stable_sig_full_model_config v2-inference.yaml \
  # --stable_sig_full_model_ckpt v2-1_512-ema-pruned.ckpt \
  # --start 0 --end 5000 \
  # --with_tracking \
  # --w_radius 10 \
  # --run_generation \
  # --additional_metrics \
  # --run_no_w \
  # --image_folder fid_eval_gt_metr_pp \
  # --msg_type binary \
  # --use_random_msgs \
  # --msg_scaler 100 \
  # --use_stable_sig

S=(
    60
    70
    80
    90
    100
)

messages=(
  "1111111111"
  "0000000000"
  "1010101010"
  "1100110011"
  "0011001100"
  "1111100000"
  "0000011111"
)

for ((j=0; j<${#messages[@]}; j++)); do
    for ((i=0; i<${#S[@]}; i++)); do
        accelerate launch -m metr.run_metr_fid \
            --project_name for_paper_tests \
            --run_name ${S[i]}_${messages[j]} --w_channel 3 --w_pattern ring \
            --start 0 --end 5000 \
            --with_tracking \
            --w_radius 10 \
            --run_generation \
            --additional_metrics \
            --run_no_w \
            --image_folder worst_message/gen_${S[i]}_${messages[j]}  \
            --msg_type binary \
            --msg ${messages[j]} \
            --msg_scaler ${S[i]} \
            --target_clean_generated
    done
done
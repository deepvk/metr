attacks=(
    "--jpeg_ratio 25"
    "--crop_scale 0.75 --crop_ratio 0.75"
    "--gaussian_blur_r 4"
    "--gaussian_std 0.1"
    "--brightness_factor 6"
    "--r_degree 75"
)

names=(
    "jpeg"
    "crop"
    "blur"
    "noise"
    "brightness"
    "rotate"

)

for ((i=0; i<${#attacks[@]}; i++)); do
    accelerate launch -m metr.run_metr \
      --project_name detect_msg_all_att_no_vae \
      --run_name ${names[i]} --w_channel 3 --w_pattern ring \
      --start 0 --end 1000 \
      --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
      --with_tracking \
      --w_radius 10 \
      --msg_type binary \
      --use_random_msgs \
      --msg_scaler 100 \
      --no_stable_sig \
      ${attacks[i]}
done

accelerate launch -m metr.run_metr \
  --project_name detect_msg_all_att_no_vae \
  --run_name diff_150 --w_channel 3 --w_pattern ring \
  --start 0 --end 1000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --w_radius 10 \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100 \
  --use_attack \
  --attack_type diff \
  --no_stable_sig \
  --diff_attack_steps 150

accelerate launch -m metr.run_metr \
  --project_name detect_msg_all_att_no_vae \
  --run_name vae_2018_q_1 --w_channel 3 --w_pattern ring \
  --start 0 --end 1000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --w_radius 10 \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100 \
  --use_attack \
  --attack_type vae \
  --vae_attack_name bmshj2018-factorized \
  --no_stable_sig \
  --vae_attack_quality 1 

accelerate launch -m metr.run_metr \
  --project_name detect_msg_all_att_no_vae \
  --run_name no_attack --w_channel 3 --w_pattern ring \
  --start 0 --end 1000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --w_radius 10 \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100 \
  --no_stable_sig \

# ----------------

bash /data/varlamov_a_data/tree-ring-watermark/scripts/stable_sig_w_metr_attacks/generate_part2.sh
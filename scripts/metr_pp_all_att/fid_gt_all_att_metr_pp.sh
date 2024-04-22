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
    accelerate launch -m metr.run_metr_fid \
      --project_name fid_gt_msg_all_att_vae \
      --run_name ${names[i]} --w_channel 3 --w_pattern ring \
      --start 0 --end 5000 \
      --with_tracking \
      --w_radius 10 \
      --run_generation \
      --additional_metrics \
      --run_no_w \
      --image_folder msg_fid_gt_all_att_vae/${names[i]} \
      --msg_type binary \
      --use_random_msgs \
      --msg_scaler 100 \
      --use_stable_sig \
      ${attacks[i]}
done

accelerate launch -m metr.run_metr_fid \
  --project_name fid_gt_msg_all_att_vae \
  --run_name diff_150 --w_channel 3 --w_pattern ring \
  --start 0 --end 5000 \
  --with_tracking \
  --w_radius 10 \
  --run_generation \
  --additional_metrics \
  --run_no_w \
  --image_folder msg_fid_gt_all_att_vae/diff_150  \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100 \
  --use_attack \
  --attack_type diff \
  --use_stable_sig \
  --diff_attack_steps 150

accelerate launch -m metr.run_metr_fid \
  --project_name fid_gt_msg_all_att_vae \
  --run_name vae_2018_q_1 --w_channel 3 --w_pattern ring \
  --start 0 --end 5000 \
  --with_tracking \
  --w_radius 10 \
  --run_generation \
  --additional_metrics \
  --run_no_w \
  --image_folder msg_fid_gt_all_att_vae/vae_2018_q_1  \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler 100 \
  --use_attack \
  --attack_type vae \
  --vae_attack_name bmshj2018-factorized \
  --use_stable_sig \
  --vae_attack_quality 1 

accelerate launch -m metr.run_metr_fid \
  --project_name fid_gt_msg_all_att_vae \
  --run_name no_attack --w_channel 3 --w_pattern ring \
  --start 0 --end 5000 \
  --with_tracking \
  --w_radius 10 \
  --run_generation \
  --additional_metrics \
  --run_no_w \
  --image_folder msg_fid_gt_all_att_vae/no_attack3  \
  --msg_type binary \
  --use_random_msgs \
  --use_stable_sig \
  --msg_scaler 100 
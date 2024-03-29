START=10
END=200
STEP_SIZE=10

for ((STEPS=START; STEPS<=END; STEPS+=STEP_SIZE)); do
    accelerate launch -m tree_ring_watermark.run_stable_tree \
      --project_name diff_attacks_metr \
      --run_name $STEPS --w_channel 3 --w_pattern ring \
      --start 0 --end 200 \
      --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
      --with_tracking \
      --w_radius 10 \
      --msg_type binary \
      --use_random_msgs \
      --msg_scaler 100 \
      --use_attack \
      --attack_type diff \
      --diff_attack_steps $STEPS \
      --no_stable_sig
done
for QUALITY in 1 2 3 4 5 6 7 8; do
  accelerate launch -m tree_ring_watermark.run_stable_tree \
    --project_name vae_attacks_metr \
    --run_name $QUALITY --w_channel 3 --w_pattern ring \
    --start 0 --end 200 \
    --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
    --with_tracking \
    --w_radius 10 \
    --msg_type binary \
    --use_random_msgs \
    --msg_scaler 100 \
    --use_attack \
    --attack_type vae \
    --vae_attack_name bmshj2018-factorized \
    --vae_attack_quality $QUALITY \
    --no_stable_sig

done
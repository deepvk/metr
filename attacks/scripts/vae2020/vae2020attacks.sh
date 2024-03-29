for QUALITY in 1 2 3 4 5 6; do
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_attack \
    --project_name new_vae_2020_attacks \
    --run_name q_$QUALITY --w_channel 3 --w_pattern ring --start 0 --end 500 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
    --with_tracking \
    --use_attack --attack_type vae --vae_attack_name cheng2020-anchor \
    --vae_attack_quality $QUALITY

done
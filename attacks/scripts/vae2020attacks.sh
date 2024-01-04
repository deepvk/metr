for QUALITY in 1 3 6 9 12 15 20
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_attack \
    --project_name vae_2020_attacks \
    --run_name vae_2020_$QUALITY --w_channel 3 --w_pattern ring --start 0 --end 200 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
    --with_tracking \
    --use_attack --attack_type vae --vae_attack_name cheng2020-anchor \
    --vae_attack_quality $QUALITY
done
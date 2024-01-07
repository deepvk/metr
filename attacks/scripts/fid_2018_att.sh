for QUALITY in 1 2 3 4 5 6 7 8; do  
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
    --project_name vae2018_metrics_gt \
    --run_name q_$QUALITY --w_channel 3 --w_pattern ring \
    --with_tracking \
    --use_attack --attack_type vae --vae_attack_name bmshj2018-factorized \
    --vae_attack_quality $QUALITY \
    --run_no_w \
    --additional_metrics
done

for QUALITY in 1 2 3 4 5 6 7 8; do  
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
    --project_name vae2018_metrics_gen \
    --run_name q_$QUALITY --w_channel 3 --w_pattern ring \
    --with_tracking \
    --use_attack --attack_type vae --vae_attack_name bmshj2018-factorized \
    --vae_attack_quality $QUALITY \
    --run_no_w \
    --additional_metrics \
    --target_clean_generated
done
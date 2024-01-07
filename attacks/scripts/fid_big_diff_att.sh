STEPS=10
END=500
STEP_SIZE=10

for ((STEPS; STEPS<=END; STEPS+=STEP_SIZE)); do
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
    --project_name big_diff_metrics_gt \
    --run_name s_$STEPS --w_channel 3 --w_pattern ring \
    --with_tracking \
    --use_attack --attack_type diff \
    --diff_attack_steps $STEPS \
    --run_no_w \
    --additional_metrics
  
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
    --project_name big_diff_metrics_gen \
    --run_name s_$STEPS --w_channel 3 --w_pattern ring \
    --with_tracking \
    --use_attack --attack_type diff \
    --diff_attack_steps $STEPS \
    --run_no_w \
    --additional_metrics \
    --target_clean_generated
done
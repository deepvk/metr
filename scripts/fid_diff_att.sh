for STEPS in 10 30 60 100 150 200 300; do  
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
    --project_name diff_metrics_gt \
    --run_name s_$STEPS --w_channel 3 --w_pattern ring \
    --with_tracking \
    --use_attack --attack_type diff \
    --diff_attack_steps $STEPS \
    --run_no_w \
    --additional_metrics
done

for STEPS in 10 30 60 100 150 200 300; do   
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
    --project_name diff_metrics_gen \
    --run_name s_$STEPS --w_channel 3 --w_pattern ring \
    --with_tracking \
    --use_attack --attack_type diff \
    --diff_attack_steps $STEPS \
    --run_no_w \
    --additional_metrics \
    --target_clean_generated
done
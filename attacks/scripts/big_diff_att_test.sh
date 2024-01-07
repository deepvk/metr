STEPS=10
END=500
STEP_SIZE=20

for ((STEPS; STEPS<=END; STEPS+=STEP_SIZE)); do
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_attack \
    --project_name big_diff_attacks \
    --run_name s_$STEPS --w_channel 3 --w_pattern ring --start 0 --end 1000 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
    --with_tracking \
    --use_attack --attack_type diff \
    --diff_attack_steps $STEPS
done
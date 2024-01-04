for STEPS in 10 30 60 100 150 200; do
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_attack \
    --project_name diff_attacks \
    --run_name diff_$STEPS --w_channel 3 --w_pattern ring --start 0 --end 200 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
    --with_tracking \
    --use_attack --attack_type diff \
    --diff_attack_steps $STEPS
  
  accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_attack \
    --project_name diff_attacks \
    --run_name diff_prompt_$STEPS --w_channel 3 --w_pattern ring --start 0 --end 200 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
    --with_tracking \
    --use_attack --attack_type diff \
    --diff_attack_steps $STEPS --use_attack_prompt

done
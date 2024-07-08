for ((steps = 10; steps <= 500; steps += 20)); do
    accelerate launch -m metr.run_metr \
    --project_name metr_diff_att \
    --run_name $steps --w_channel 3 --w_pattern ring \
    --start 0 --end 1000 \
    --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
    --with_tracking \
    --w_radius 10 \
    --msg_type binary \
    --use_random_msgs \
    --msg_scaler 100 \
    --use_attack \
    --attack_type diff \
    --no_stable_sig \
    --diff_attack_steps $steps
done
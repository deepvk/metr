# accelerate launch -m metr.run_metr \
#   --project_name for_paper_tests \
#   --run_name no_attack --w_channel 3 --w_pattern ring \
#   --start 0 --end 1000 \
#   --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
#   --with_tracking \
#   --msg_type binary \
#   --use_random_msgs \
#   --save_locally \
#   --w_radius 10 \
#   --msg_scaler 100 \
#   # --no_stable_sig \
#   --local_path generated_images
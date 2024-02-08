accelerate launch -m tree_ring_watermark.run_stable_tree \
  --project_name stable_tree \
  --run_name first_test --w_channel 3 --w_pattern ring \
  --start 0 --end 1000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --save_locally \
  --local_path /data/varlamov_a_data/dima/imgs_stable_tree